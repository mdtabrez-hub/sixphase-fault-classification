#!/usr/bin/env python3
"""
spemt.py -- batched six-phase electromagnetic-transient solver, in Python.

MODEL
-----
Constant-parameter distributed line (Bergeron / Dommel travelling wave) with
the series resistance of each section lumped as R/2 at each end.  For mode m
of a section of length s:

    Zc = sqrt(l/c)      tau = s sqrt(l c)      R = r s
    Z' = Zc + R/2       h   = (Zc - R/2)/(Zc + R/2)

    i_A(t) = v_A(t)/Z' - b_B(t - tau)          b_A(t) = v_A(t)/Z' + h i_A(t)

Modal quantities map to the phase domain through the real constant-parameter
transformation of spconst.modal_transform, so each terminal stamps a constant
6x6 admittance Ti diag(1/Z') Tv^-1 in parallel with a history current source.

NODES (25 per case)
    0.. 5 bus S (relay)   6..11 fault node line-1 side (F1)
   12..17 fault node line-2 side (F2)      18..23 bus R      24 fault star N

F1-F2 are joined conductor by conductor through a series switch, so an open
conductor is represented by opening that switch.  A shunt fault connects the
selected conductors of F1 to N and N to earth.

Every case carries a pre-factorised admittance inverse per switching state.
The batch is advanced together.

A matched FREQUENCY-DOMAIN twin (phasor_solve) of exactly the same network is
provided.  It is used (a) to initialise every time-domain run from the phasor
solution, as the manuscript's validation protocol requires, and (b) as an
independent check on the time-domain solver.  An 'exact' mode solves the same
network with the exact distributed two-port instead of the Bergeron
approximation, so that the modelling error of the line representation itself
can be quantified.

Nothing here is transcribed from an earlier result.
"""
import numpy as np
import spconst as sc

TWO_PI = 2.0 * np.pi
NN = 25
S_, F1_, F2_, R_ = slice(0, 6), slice(6, 12), slice(12, 18), slice(18, 24)
N_ = 24
G_CLOSED = 1.0e4        # S, closed series switch (0.1 mOhm)
G_OPEN   = 1.0e-9       # S, open series switch
R_BOLT   = 1.0e-4       # ohm, bolted connection


def rl_companion(Rm, Lm, dt):
    """Trapezoidal companion of a series R-L matrix branch:
       i_t = Y (e_t + e_{t-1} - v_t - v_{t-1}) + Y M i_{t-1}."""
    A = 2.0 * Lm / dt
    Y = np.linalg.inv(A + Rm)
    return Y, (A - Rm)


# ===========================================================================
class Network:
    """Per-case network description shared by the time- and frequency-domain
    solvers."""

    def __init__(self, spec, table=None, lc=None, md=None):
        self.t = table or sc.TABLE_I
        self.lc = lc if lc is not None else sc.line_constants(self.t)
        self.md = md if md is not None else sc.modal_transform(self.lc, self.t)
        self.spec = spec
        self.B = len(spec["d_km"])
        t, md = self.t, self.md
        L = t["length_km"]
        d = np.asarray(spec["d_km"], float)
        self.seg = np.stack([d, L - d], axis=1)                     # (B,2) km
        self.Rsec = self.seg[:, :, None] * md["r"][None, None, :]   # (B,2,6)
        self.tau = self.seg[:, :, None] * md["tau_per_km"][None, None, :]
        self.Zc = md["Zc"]
        w = TWO_PI * t["f0"]
        self.w = w
        # source / remote / load impedance matrices
        self.Zs = np.stack([sc.source_impedance(ss, xr, t["Zt_pct"], t["S_T_MVA"],
                                                t["Rw_pu"], zn, t)
                            for ss, xr, zn in zip(spec["S_sc"], spec["XR"], spec["Zn_ohm"])])
        # remote equivalent source: present only in the IBR studies, where the
        # synchronous share of the remote short-circuit contribution is (1-pen)
        pen = np.asarray(spec.get("ibr_pen", np.zeros(self.B)), float)
        has_rem = np.asarray(spec.get("remote_source", np.zeros(self.B, bool)), bool)
        ssR = np.where(has_rem, t["S_scR_MVA"] * np.maximum(1.0 - pen, 1e-3), 1e-6)
        self.ZsR = np.stack([sc.source_impedance(s_, t["XR_remote"], t["Zt_pct"],
                                                 t["S_T_MVA"], t["Rw_pu"], 0.0, t)
                             for s_ in ssR])
        self.Zl = np.stack([sc.load_impedance(p, t) for p in spec["load_pu"]])
        lp2 = spec.get("load_pu2")
        self.Zl2 = (None if lp2 is None else
                    np.stack([sc.load_impedance(p, t) for p in lp2]))
        self.slip = np.asarray(spec.get("slip_hz", np.zeros(self.B)), float)
        Vpk, ang = sc.source_emf(t)
        self.Vpk, self.ang = Vpk, ang
        self.th0 = np.deg2rad(np.asarray(spec.get("inception_deg", np.zeros(self.B)), float))
        self.ph0 = self.th0[:, None] + ang[None, :]

    # ---------------------------------------------------------- fault stamps
    def fault_stamp(self, G, mask, grounded, Rf, open_mask):
        """Add the series switches and the shunt fault to a (B,25,25) matrix."""
        B = self.B
        gser = np.where(open_mask, G_OPEN, G_CLOSED)
        for i in range(6):
            G[:, 6 + i, 6 + i] += gser[:, i]
            G[:, 12 + i, 12 + i] += gser[:, i]
            G[:, 6 + i, 12 + i] -= gser[:, i]
            G[:, 12 + i, 6 + i] -= gser[:, i]
        Rf = np.maximum(np.asarray(Rf, float), R_BOLT)
        Ra = np.where(grounded[:, None], R_BOLT, 0.5 * Rf[:, None])
        Rg = np.where(grounded, Rf, 1e12)
        ga = np.where(mask, 1.0 / Ra, 0.0)
        gg = 1.0 / Rg
        for i in range(6):
            G[:, 6 + i, 6 + i] += ga[:, i]
            G[:, N_, N_] += ga[:, i]
            G[:, 6 + i, N_] -= ga[:, i]
            G[:, N_, 6 + i] -= ga[:, i]
        G[:, N_, N_] += gg + 1e-12
        return G, gg

    def states(self):
        """List of (mask, grounded, Rf, open_mask) switching states."""
        sp, B = self.spec, self.B
        st = [dict(mask=np.zeros((B, 6), bool), grounded=np.zeros(B, bool),
                   Rf=np.full(B, 1e9), open_mask=np.zeros((B, 6), bool))]
        st.append(dict(mask=np.asarray(sp["mask"], bool),
                       grounded=np.asarray(sp["grounded"], bool),
                       Rf=np.asarray(sp["Rf_ohm"], float),
                       open_mask=np.asarray(sp["open_mask"], bool)))
        if "mask2" in sp:
            st.append(dict(mask=np.asarray(sp["mask2"], bool),
                           grounded=np.asarray(sp["grounded2"], bool),
                           Rf=np.asarray(sp["Rf_ohm"], float),
                           open_mask=np.asarray(sp["open_mask2"], bool)
                           if "open_mask2" in sp else np.asarray(sp["open_mask"], bool)))
        return st

    # -------------------------------------------------------- frequency dom.
    def line_twoport(self, dt=None, kind="bergeron"):
        """Modal (Y11, Y12) of each line section at f0.  (B,2,6) complex."""
        w, md = self.w, self.md
        if kind == "bergeron":
            Zp = self.Zc[None, None, :] + 0.5 * self.Rsec
            h = (self.Zc[None, None, :] - 0.5 * self.Rsec) / Zp
            x = np.exp(-1j * w * self.tau)
            den = Zp * (1.0 - (x ** 2) * (h ** 2))
            Y11 = (1.0 + (x ** 2) * h) / den
            Y12 = -x * (1.0 + h) / den
        else:                                     # exact distributed two-port
            z = md["r"] + 1j * w * md["l"]        # per km
            y = 1j * w * md["c"]
            gam = np.sqrt(z * y)[None, None, :]
            Yc = np.sqrt(y / z)[None, None, :]
            gs = gam * self.seg[:, :, None]
            Y11 = Yc / np.tanh(gs)
            Y12 = -Yc / np.sinh(gs)
        return Y11, Y12

    def phasor_solve(self, state, kind="bergeron"):
        """Steady-state phasor solution of the whole network (peak-referenced
        complex amplitudes).  Returns node voltages (B,25) complex."""
        B, md = self.B, self.md
        Ti, Tvi = md["Ti"], md["Tv_inv"]
        Y11, Y12 = self.line_twoport(kind=kind)
        G = np.zeros((B, NN, NN), complex)

        def blk(Ym):                                     # (B,6) modal -> (B,6,6)
            return np.einsum('ik,bk,kj->bij', Ti, Ym, Tvi)

        for s, (a, b) in enumerate([(S_, F1_), (F2_, R_)]):
            G[:, a, a] += blk(Y11[:, s]); G[:, b, b] += blk(Y11[:, s])
            G[:, a, b] += blk(Y12[:, s]); G[:, b, a] += blk(Y12[:, s])
        Ys = np.linalg.inv(self.Zs); YsR = np.linalg.inv(self.ZsR)
        Yl = np.linalg.inv(self.Zl)
        G[:, S_, S_] += Ys
        G[:, R_, R_] += YsR + Yl
        Gr = np.zeros((B, NN, NN))
        Gr, _ = self.fault_stamp(Gr, state["mask"], state["grounded"],
                                 state["Rf"], state["open_mask"])
        G = G + Gr
        E = self.Vpk * np.exp(1j * self.ph0)             # (B,6) peak phasors
        J = np.zeros((B, NN), complex)
        J[:, S_] = np.einsum('bij,bj->bi', Ys, E)
        J[:, R_] = np.einsum('bij,bj->bi', YsR, E)
        if getattr(self, "_ibr_ready", False):
            # grid-following: a current source of the declared rating.
            # grid-forming: an e.m.f. behind the virtual impedance Zv.
            Yv = np.linalg.inv(self.Zv)
            G[:, R_, R_] += Yv * self.gfm[:, None, None]
            J[:, R_] += self.I_gfl
            J[:, R_] += np.einsum('bij,bj->bi', Yv, self.E_gfm) * self.gfm[:, None]
        return np.linalg.solve(G, J[..., None])[..., 0]


# ===========================================================================
class Batch(Network):
    """Time-domain solver."""

    def __init__(self, spec, dt, t_end, t_fault, table=None, lc=None, md=None,
                 arc=False, ibr=False, seed=0):
        self.arc = bool(arc); self.ibr = bool(ibr); self.seed = int(seed)
        h = spec.get("harm_A")
        self.harm = None if h is None or not np.any(h) else np.asarray(h, float)
        super().__init__(spec, table=table, lc=lc, md=md)
        self.dt = float(dt)
        self.n = int(round(t_end / dt))
        self.t_fault = float(t_fault)
        md = self.md
        self.Zp = self.Zc[None, None, :] + 0.5 * self.Rsec
        self.h = (self.Zc[None, None, :] - 0.5 * self.Rsec) / self.Zp
        idxf = self.tau / self.dt
        self.kint = np.floor(idxf).astype(np.int64)
        self.frac = idxf - self.kint
        if self.kint.min() < 1:
            raise ValueError(f"tau/dt = {idxf.min():.3f} < 1: shorten dt")
        self.nbuf = int(self.kint.max()) + 3
        self.Yc = np.einsum('ik,bsk,kj->bsij', md["Ti"], 1.0 / self.Zp, md["Tv_inv"])
        self.Ys, self.Ms = rl_companion(self.Zs.real, self.Zs.imag / self.w, dt)
        self.YsR, self.MsR = rl_companion(self.ZsR.real, self.ZsR.imag / self.w, dt)
        self.Yl, self.Ml = rl_companion(self.Zl.real, self.Zl.imag / self.w, dt)
        if self.Zl2 is not None:
            self.Yl2, self.Ml2 = rl_companion(self.Zl2.real, self.Zl2.imag / self.w, dt)
        else:
            self.Yl2, self.Ml2 = self.Yl, self.Ml
        self.stlist = self.states()
        if self.arc:                       # ground path replaced by the arc branch
            for st in self.stlist[1:]:
                st["Rf"] = np.full(self.B, 1e12)
        # the grid-forming converter is a voltage source behind a virtual
        # impedance, so its admittance has to be stamped before the nodal
        # matrix is factorised
        if self.ibr:
            self._ibr_setup()
        self.Ainv = np.stack([self._inv(st, None if k == 0 else self.Yl2)
                              for k, st in enumerate(self.stlist)])
        self.ggs = np.stack([self._gg(st) for st in self.stlist])
        if self.arc:
            self._arc_setup()

    # ------------------------------------------------------------------ arc
    def _arc_setup(self):
        """Two-branch asymmetric arc, Eq. (6):
             v_fg = Rp+ i + Vs+ (1 - exp(-i/I0+))        i >= 0
                  = Rp- i - Vs- (1 - exp( i/I0-))        i <  0
        Path resistances are resampled independently for each half-cycle
        polarity every t_res seconds (Section 2.5)."""
        sp, B = self.spec, self.B
        self.arc_p = dict(
            Vsp=np.asarray(sp.get("Vs_plus", np.full(B, 6.15e3)), float),
            Vsm=np.asarray(sp.get("Vs_minus", np.full(B, 5.50e3)), float),
            I0p=np.asarray(sp.get("I0_plus", np.full(B, 10.0)), float),
            I0m=np.asarray(sp.get("I0_minus", np.full(B, 8.0)), float),
            Rmin=np.asarray(sp.get("Rp_min", np.full(B, 2.5e3)), float),
            Rmax=np.asarray(sp.get("Rp_max", np.full(B, 6.0e3)), float))
        self.t_res = float(sp.get("t_resample", 1.0e-4))
        nres = int(np.ceil(self.n * self.dt / self.t_res)) + 2
        rng = np.random.default_rng([self.seed, 77])
        lo, hi = self.arc_p["Rmin"], self.arc_p["Rmax"]
        self.Rp_seq = lo[None, :, None] + (hi - lo)[None, :, None] * rng.random((nres, B, 2))
        self.nres = nres

    def _arc_current(self, Vth, Zth, Rp_p, Rp_m):
        """Solve  Vth - Zth i = v_fg(i)  by Newton (monotone, 6 iterations)."""
        p = self.arc_p
        i = Vth / (Zth + np.where(Vth >= 0, Rp_p, Rp_m))
        for _ in range(6):
            pos = i >= 0
            Rp = np.where(pos, Rp_p, Rp_m)
            Vs = np.where(pos, p["Vsp"], -p["Vsm"])
            I0 = np.where(pos, p["I0p"], p["I0m"])
            ex = np.exp(-np.abs(i) / I0)
            v = Rp * i + Vs * (1.0 - ex)
            dv = Rp + np.abs(Vs) / I0 * ex
            f = Vth - Zth * i - v
            i = i + f / (Zth + dv)
        return i

    # ------------------------------------------------------------------ IBR
    def _ibr_setup(self):
        """Averaged-value converter at the remote terminal, Table XI.

        Two control modes are represented and they are genuinely different
        network elements, not two labels for one:

          grid-following (mode 0) -- a current source whose magnitude is set by
              the converter rating and whose angle follows an SRF-PLL, with
              reactive-current priority on voltage depression (IEEE Std 2800).
              It presents no admittance to the network.

          grid-forming  (mode 1) -- an internal e.m.f. behind a virtual
              impedance Zv (0.20 pu on the converter rating, X/R = 5, zero
              sequence open because the converter interface is not earthed).
              It therefore DOES present an admittance, which is stamped into
              the nodal matrix, and its fault current is set by the network
              rather than by a reference.  The current limit is enforced by
              reducing the internal e.m.f. through a first-order lag of 2 ms,
              which is the voltage-magnitude limiting strategy; the virtual
              impedance itself is held constant so that the nodal matrix is
              still factorised once per switching state.

        The internal e.m.f. and the grid-following current reference are both
        derived from a phasor solution of the network WITHOUT the converter, so
        that the two modes deliver the same pre-fault output and the comparison
        isolates the fault response.  ADDED: the manuscript declares the control
        modes in Table X and the converter in Table XI, but declares neither the
        virtual impedance nor the limiting strategy; both are stated here.
        """
        t, B, sp = self.t, self.B, self.spec
        b = sc.bases(t)
        pen = np.asarray(sp.get("ibr_pen", np.zeros(B)), float)
        # converter rating from its share of the remote short-circuit contribution
        S_conv = pen * t["S_scR_MVA"] * 1e6 / 20.0        # SCR-consistent rating
        self.I_rated = np.maximum(S_conv / (6.0 * b["V_base"]), 1e-6)   # A rms/phase
        self.ibr_mode = np.asarray(sp.get("ibr_mode", np.zeros(B, int)), int)  # 0=GFL 1=GFM
        self.Ilim1 = 1.2 * self.I_rated
        self.Ilim2 = 1.5 * self.I_rated
        self.Vnom = b["V_base"]
        self.pll_wn = 2 * np.pi * 20.0
        act = pen > 0
        self.gfm = ((self.ibr_mode == 1) & act).astype(float)
        self.gfl = ((self.ibr_mode == 0) & act).astype(float)
        # virtual impedance, zero sequence open
        Zb_c = self.Vnom / self.I_rated
        Xv = 0.20 * Zb_c
        zv = Xv / 5.0 + 1j * Xv
        self.Zv = np.stack([sc.seq_to_phase(np.array([1e9 + 0j, z, z, z, z, z]))
                            for z in zv])
        # steady state WITHOUT the converter, to set both references
        V0 = self.phasor_solve(self.stlist[0], kind="bergeron")
        VR0 = V0[:, R_]
        Iref = (np.sqrt(2.0) * self.I_rated[:, None]
                * np.exp(1j * np.angle(VR0)) * act[:, None])
        self.I_gfl = Iref * self.gfl[:, None]
        self.E_gfm = VR0 + np.einsum('bij,bj->bi', self.Zv, Iref)
        self.Yv_c, self.Mv_c = rl_companion(self.Zv.real, self.Zv.imag / self.w,
                                            self.dt)
        self.Yv_c = self.Yv_c * self.gfm[:, None, None]
        self.YvM_c = self.Yv_c @ self.Mv_c
        self._has_gfm = bool(self.gfm.any())
        self._ibr_ready = True
        # one refinement pass: the references above were set on a network
        # without the converter, so re-solve WITH it and re-align both the
        # injection angle and the phase-locked loop to that solution.  Without
        # this the loop starts a fraction of a degree out of lock and rings at
        # the loop bandwidth through the whole pre-fault window.
        for _ in range(3):
            V1 = self.phasor_solve(self.stlist[0], kind="bergeron")
            VR1 = V1[:, R_]
            Iref = (np.sqrt(2.0) * self.I_rated[:, None]
                    * np.exp(1j * np.angle(VR1)) * act[:, None])
            self.I_gfl = Iref * self.gfl[:, None]
            self.E_gfm = VR1 + np.einsum('bij,bj->bi', self.Zv, Iref)
        V1 = self.phasor_solve(self.stlist[0], kind="bergeron")
        VR1 = V1[:, R_]
        ang6 = -TWO_PI / 6.0 * np.arange(6)
        A = np.sqrt(2.0 / 6.0) * (VR1 * np.cos(ang6)[None, :]).sum(1)
        Bv = -np.sqrt(2.0 / 6.0) * (VR1 * np.sin(ang6)[None, :]).sum(1)
        self._pll0 = np.arctan2(Bv.real, A.real)
        self._vpu0 = np.clip(np.hypot(np.abs(A), np.abs(Bv))
                             / (np.sqrt(6.0) * self.Vnom), 0.0, 2.0)

    def _base_G(self, Yl=None):
        B = self.B
        G = np.zeros((B, NN, NN))
        G[:, S_, S_] += self.Yc[:, 0]; G[:, F1_, F1_] += self.Yc[:, 0]
        G[:, F2_, F2_] += self.Yc[:, 1]; G[:, R_, R_] += self.Yc[:, 1]
        G[:, S_, S_] += self.Ys
        G[:, R_, R_] += self.YsR + (self.Yl if Yl is None else Yl)
        if getattr(self, "_ibr_ready", False):
            G[:, R_, R_] += self.Yv_c
        return G

    def _inv(self, st, Yl=None):
        G, _ = self.fault_stamp(self._base_G(Yl), st["mask"], st["grounded"],
                                st["Rf"], st["open_mask"])
        return np.linalg.inv(G)

    def _gg(self, st):
        Rg = np.where(st["grounded"], np.maximum(st["Rf"], R_BOLT), 1e12)
        return 1.0 / Rg

    # ------------------------------------------------------------------ run
    def run(self, decim=20, init="phasor", relay_at="S"):
        B, dt, n, nb = self.B, self.dt, self.n, self.nbuf
        md = self.md
        Ti, Tvi, Tii = md["Ti"], md["Tv_inv"], md["Ti_inv"]
        Zp, h, Yc = self.Zp, self.h, self.Yc
        ns = len(self.stlist)
        w = self.w
        YsM = self.Ys @ self.Ms
        YsRM = self.YsR @ self.MsR
        YlM = self.Yl @ self.Ml
        YlM2 = self.Yl2 @ self.Ml2
        self._has_slip = bool(np.any(self.slip))
        buf = np.zeros((nb, B, 2, 2, 6))
        i_src = np.zeros((B, 6)); v_s_p = np.zeros((B, 6)); e_p = np.zeros((B, 6))
        i_srcR = np.zeros((B, 6)); v_r_p = np.zeros((B, 6)); i_ld = np.zeros((B, 6))
        e_pR = np.zeros((B, 6))
        gfm_on = bool(getattr(self, "_has_gfm", False))
        if self.ibr and not hasattr(self, "_ang6"):
            self._ang6 = -TWO_PI / 6.0 * np.arange(6)
        if gfm_on:
            Emag = np.abs(self.E_gfm); Eang = np.angle(self.E_gfm)
            i_v = np.zeros((B, 6)); e_v_p = np.zeros((B, 6))
            s_gfm = np.ones(B)                 # e.m.f. scaling of the limiter
            tau_lim = 2.0e-3
            YvM = self.YvM_c
        if init == "phasor":
            o = {}
            self._init_phasor(buf, o)
            i_src, v_s_p, e_p = o["i_src"], o["v_s_p"], o["e_p"]
            i_srcR, v_r_p, i_ld = o["i_srcR"], o["v_r_p"], o["i_ld"]
            e_pR = e_p.copy()
            if gfm_on:
                i_v = o["i_v"]; e_v_p = o["e_v_p"]
        nout = n // decim
        out_v = np.zeros((nout, B, 6), np.float32)
        out_i = np.zeros((nout, B, 6), np.float32)
        out_if = np.zeros((nout, B), np.float32)
        t_evolve = np.asarray(self.spec.get("t_evolve", np.full(B, np.inf)), float)
        ba = np.arange(B)[:, None, None, None]
        sa = np.arange(2)[None, :, None, None]
        ee = np.array([1, 0])[None, None, :, None]
        ma = np.arange(6)[None, None, None, :]
        arB = np.arange(B)
        k4 = self.kint[:, :, None, :]                       # (B,2,1,6)
        f4 = self.frac[:, :, None, :]
        Zp4 = Zp[:, :, None, :]; h4 = h[:, :, None, :]
        Ain = self.Ainv
        vt = np.empty((B, 2, 2, 6))
        rhs = np.zeros((B, NN))
        sid_prev = None; Ai = None
        for step in range(n):
            tn = step * dt
            if ns > 2:
                sid = np.where(tn >= t_evolve, 2, np.where(tn >= self.t_fault, 1, 0))
            else:
                sid = np.full(B, 1 if tn >= self.t_fault else 0, np.int64)
            if sid_prev is None or not np.array_equal(sid, sid_prev):
                Ai = np.ascontiguousarray(Ain[sid, arB]); sid_prev = sid.copy()
            e = self.Vpk * np.cos(w * tn + self.ph0)
            eR = e if not self._has_slip else self.Vpk * np.cos(
                (w + 2 * np.pi * self.slip[:, None]) * tn + self.ph0)
            if self.harm is not None:
                ih = (self.harm[:, None] *
                      (0.6 * np.cos(5 * w * tn + 5 * self.ph0)
                       + 0.4 * np.cos(7 * w * tn + 7 * self.ph0)))
            else:
                ih = None
            i0 = (step - k4) % nb
            i1 = (step - k4 - 1) % nb
            far = buf[i0, ba, sa, ee, ma]
            far += f4 * (buf[i1, ba, sa, ee, ma] - far)
            J = np.einsum('ij,bsej->bsei', Ti, far)
            eep = e + e_p
            if ih is not None:
                rhs[:, R_] = J[:, 1, 1] + ih
            rhs[:, S_] = J[:, 0, 0] + (eep - v_s_p) @ self.Ys.T if False else 0.0
            if ih is not None:
                rhs[:, R_] = J[:, 1, 1] + ih
            rhs[:, S_] = J[:, 0, 0]
            rhs[:, F1_] = J[:, 0, 1]; rhs[:, F2_] = J[:, 1, 0]
            rhs[:, R_] = J[:, 1, 1] + (0.0 if ih is None else ih)
            rhs[:, S_] += np.einsum('bij,bj->bi', self.Ys, eep - v_s_p) \
                        + np.einsum('bij,bj->bi', YsM, i_src)
            eepR = eR + e_pR
            Ylx, YlMx = (self.Yl, YlM) if tn < self.t_fault else (self.Yl2, YlM2)
            rhs[:, R_] += np.einsum('bij,bj->bi', self.YsR, eepR - v_r_p) \
                        + np.einsum('bij,bj->bi', YsRM, i_srcR) \
                        - np.einsum('bij,bj->bi', Ylx, v_r_p) \
                        - np.einsum('bij,bj->bi', YlMx, i_ld)
            if self.ibr:
                rhs[:, R_] += self._ibr_inject(step, tn, v_r_p)
                if gfm_on:
                    e_v = s_gfm[:, None] * Emag * np.cos(w * tn + Eang)
                    rhs[:, R_] += np.einsum('bij,bj->bi', self.Yv_c,
                                            e_v + e_v_p - v_r_p) \
                                + np.einsum('bij,bj->bi', YvM, i_v)
                v = np.einsum('bij,bj->bi', Ai, rhs)
            elif self.arc:
                v = np.einsum('bij,bj->bi', Ai, rhs)
            else:
                v = np.einsum('bij,bj->bi', Ai, rhs)
            if self.arc:
                act = (sid > 0)
                Zth = Ai[:, N_, N_]
                jr = min(int(tn / self.t_res), self.nres - 1)
                iarc = self._arc_current(v[:, N_], Zth,
                                         self.Rp_seq[jr, :, 0], self.Rp_seq[jr, :, 1])
                iarc = np.where(act, iarc, 0.0)
                v = v - Ai[:, :, N_] * iarc[:, None]
                self._iarc = iarc
            vt[:, 0, 0] = v[:, S_]; vt[:, 0, 1] = v[:, F1_]
            vt[:, 1, 0] = v[:, F2_]; vt[:, 1, 1] = v[:, R_]
            ilin = np.einsum('bsij,bsej->bsei', Yc, vt) - J
            vm = np.einsum('ij,bsej->bsei', Tvi, vt)
            vm /= Zp4
            im = np.einsum('ij,bsej->bsei', Tii, ilin)
            buf[step % nb] = vm + h4 * im
            vS = v[:, S_]; vR = v[:, R_]
            i_src = np.einsum('bij,bj->bi', self.Ys, eep - vS - v_s_p) \
                  + np.einsum('bij,bj->bi', YsM, i_src)
            i_srcR = np.einsum('bij,bj->bi', self.YsR, eepR - vR - v_r_p) \
                   + np.einsum('bij,bj->bi', YsRM, i_srcR)
            i_ld = np.einsum('bij,bj->bi', Ylx, vR + v_r_p) \
                 + np.einsum('bij,bj->bi', YlMx, i_ld)
            if gfm_on:
                i_v = np.einsum('bij,bj->bi', self.Yv_c, e_v + e_v_p - vR - v_r_p) \
                    + np.einsum('bij,bj->bi', YvM, i_v)
                # current limit by internal-e.m.f. reduction, first-order lag
                ia = np.sqrt(2.0 / 6.0) * (i_v * np.cos(self._ang6)[None, :]).sum(1)
                ib = -np.sqrt(2.0 / 6.0) * (i_v * np.sin(self._ang6)[None, :]).sum(1)
                # Clarke magnitude of a balanced set of r.m.s. current I is
                # sqrt(6) I, so the limit is compared on the same scale
                imag_pk = np.hypot(ia, ib)
                lim_pk = np.sqrt(6.0) * np.where(tn - self.t_fault < 0.100,
                                                 self.Ilim2, self.Ilim1)
                s_t = np.minimum(1.0, lim_pk / np.maximum(imag_pk, 1e-9))
                s_gfm = s_gfm + (s_t - s_gfm) * (dt / tau_lim)
                s_gfm = np.clip(s_gfm, 0.05, 1.0)
                e_v_p = e_v
            e_p = e; e_pR = eR; v_s_p = vS; v_r_p = vR
            if step % decim == 0:
                j = step // decim
                if j < nout:
                    if relay_at == "S":
                        out_v[j] = vS; out_i[j] = ilin[:, 0, 0]
                    else:                      # relay at the fault node, looking
                        out_v[j] = v[:, F2_]   # towards the remote terminal
                        out_i[j] = ilin[:, 1, 0]
                    out_if[j] = self._iarc if self.arc else self.ggs[sid, arB] * v[:, N_]
        return dict(v=out_v, i=out_i, i_fault=out_if, dt=dt * decim, n=nout)

    def _ibr_inject(self, step, tn, vR):
        """Averaged-value converter current injection at bus R.
        Grid-following: SRF-PLL, reactive-current priority per IEEE Std 2800.
        Grid-forming: voltage-source behaviour behind a virtual impedance,
        realised as a current injection proportional to the voltage error and
        clamped at the converter limit."""
        if not hasattr(self, "_pll"):
            # the phase-locked loop starts locked to the pre-fault terminal
            # voltage, so that no start-up transient enters the pre-fault window
            self._pll = self._pll0.copy()
            self._ang6 = -TWO_PI / 6.0 * np.arange(6)
        # Six-phase Clarke projection.  The conductor reference angles are
        # ang6_k = -2*pi*k/6, so the quadrature axis must be projected on
        # -ang6_k; projecting on +ang6_k returns a vector that rotates
        # BACKWARDS, which no phase-locked loop forced to +w can ever lock to.
        va = np.sqrt(2.0 / 6.0) * (vR * np.cos(self._ang6)).sum(1)
        vb = -np.sqrt(2.0 / 6.0) * (vR * np.sin(self._ang6)).sum(1)
        mag = np.hypot(va, vb)
        th = np.arctan2(vb, va)
        err = np.angle(np.exp(1j * (th - self._pll)))
        self._pll = self._pll + (self.w + self.pll_wn * err) * self.dt
        # The six-phase Clarke magnitude of a balanced set of r.m.s. conductor
        # voltage V is sqrt(6) V, so that is the divisor.  (An earlier form
        # divided by sqrt(3) V and returned 1.41 pu on a healthy line, which
        # held the reactive-current priority of Table XI permanently off.)
        vpu = np.clip(mag / (np.sqrt(6.0) * self.Vnom), 0.0, 2.0)
        # The reactive-current reference is formed on a FILTERED magnitude.
        # Feeding the instantaneous Clarke magnitude into a proportional
        # voltage-support loop closes an algebraic loop through the network on
        # the 2*omega ripple of that magnitude, and the loop diverges; real
        # ride-through controls measure over a filtered or r.m.s. magnitude and
        # IEEE Std 2800 allows tens of milliseconds to reach the reference.
        # tau_v = 5 ms.  ADDED: the manuscript declares the priority but not
        # the measurement filter.
        if not hasattr(self, "_vf"):
            self._vf = np.full(self.B, 1.0) if not hasattr(self, "_vpu0") \
                else self._vpu0.copy()
        self._vf = self._vf + (vpu - self._vf) * (self.dt / 5.0e-3)
        dV = np.clip(1.0 - self._vf, 0.0, 1.0)
        lim = np.where(tn - self.t_fault < 0.100, self.Ilim2, self.Ilim1)
        Iq = np.minimum(2.0 * dV * self.I_rated, lim)
        Id = np.sqrt(np.maximum(lim ** 2 - Iq ** 2, 0.0))
        Id = np.minimum(Id, self.I_rated)
        Imag = np.hypot(Id, Iq) * np.sqrt(2.0)
        phi = np.arctan2(Iq, np.maximum(Id, 1e-9))
        return (self.gfl[:, None] * Imag[:, None]
                * np.cos(self._pll[:, None] + self._ang6[None, :] - phi[:, None]))

    # ------------------------------------------------------- initialisation
    def _init_phasor(self, buf, out):
        """Fill history buffers and companion states from the pre-fault phasor
        solution of the identical network."""
        B, dt, nb, md = self.B, self.dt, self.nbuf, self.md
        Tvi, Tii = md["Tv_inv"], md["Ti_inv"]
        V = self.phasor_solve(self.stlist[0], kind="bergeron")        # (B,25)
        Y11, Y12 = self.line_twoport(kind="bergeron")
        Vt = np.stack([V[:, S_], V[:, F1_], V[:, F2_], V[:, R_]], 1).reshape(B, 2, 2, 6)
        Vm = np.einsum('ij,bsej->bsei', Tvi, Vt)
        Im = np.empty_like(Vm)
        Im[:, :, 0, :] = Y11 * Vm[:, :, 0, :] + Y12 * Vm[:, :, 1, :]
        Im[:, :, 1, :] = Y12 * Vm[:, :, 0, :] + Y11 * Vm[:, :, 1, :]
        Bph = Vm / self.Zp[:, :, None, :] + self.h[:, :, None, :] * Im   # complex b
        w = self.w
        for m in range(nb):
            q = (nb - m) % nb
            if q == 0:
                q = nb
            buf[m] = (Bph * np.exp(-1j * w * q * dt)).real
        E = self.Vpk * np.exp(1j * self.ph0)
        VS, VR = V[:, S_], V[:, R_]
        Ys_c = np.linalg.inv(self.Zs); YsR_c = np.linalg.inv(self.ZsR)
        Yl_c = np.linalg.inv(self.Zl)
        Isrc = np.einsum('bij,bj->bi', Ys_c, E - VS)
        IsrcR = np.einsum('bij,bj->bi', YsR_c, E - VR)
        Ild = np.einsum('bij,bj->bi', Yl_c, VR)
        ph = np.exp(-1j * w * dt)
        out["i_src"] = (Isrc * ph).real
        out["i_srcR"] = (IsrcR * ph).real
        out["i_ld"] = (Ild * ph).real
        out["v_s_p"] = (VS * ph).real
        out["v_r_p"] = (VR * ph).real
        out["e_p"] = (E * ph).real
        if getattr(self, "_ibr_ready", False) and getattr(self, "_has_gfm", False):
            Yv_c = np.linalg.inv(self.Zv)
            Iv = np.einsum('bij,bj->bi', Yv_c, self.E_gfm - VR) * self.gfm[:, None]
            out["i_v"] = (Iv * ph).real
            out["e_v_p"] = (self.E_gfm * ph).real
