#!/usr/bin/env python3
"""
sprelay.py -- the adaptive sequence-modal six-phase relay of Algorithm 1.

Implements, line for line, the relay specified in Sections 3-5 of the
manuscript: full-cycle cosine-sine phasor estimation, the power-invariant
six-phase transform (4), the 85-dimensional feature vector (11)-(14), the
normalised disturbance index (15), the adaptive threshold (16), the trip
condition (17), the high-impedance-fault index (18)-(19), the distance
projection (20), the internal-zone security gate (21) and the Huber
M-estimator locator (23)-(24).

Three quantities that the manuscript specifies only in words are given
concrete, declared definitions here and are flagged as ADDED:
  * the superimposed-quantity reference (a two-cycle memory),
  * the transient-energy base  E_base = N_w I_base^2,
  * the current-transformer health indicator eta_CT.
The asymmetry statistic (19) is evaluated on the MEASURED residual current
rather than on the fault current, which a single-ended relay cannot observe;
this correction is recorded in the manuscript.
"""
import numpy as np
from scipy import signal
import spconst as sc

SET = dict(fs=5000.0, f0=50.0, Nw=100, wI=0.50, wV=0.20, wT=0.30,
           tau0=0.05, kL=4.0, kC=0.15, kN=3.0, kR=0.03, Rf_max=1000.0,
           m_pers=4, a1=0.35, a2=0.25, a3=0.20, a4=0.20, Gam_th=0.25,
           m_HIF=3, beta=0.85, beta_rev=0.05, r_max=0.15, C_min=0.70,
           delta_H=1.345, K=3, B_boot=1000,
           s0_CT=6.0, s1_CT=25.0, mem_cycles=2, mimic=True)

DB4_H = np.array([-0.23037781, 0.71484657, -0.63088077, -0.02798377,
                   0.18703481, 0.03084138, -0.03288301, -0.01059740])[::-1]


# ----------------------------------------------------------------- phasors
def dft_phasors(x, Nw, harmonic=1):
    """Sliding full-cycle cosine-sine DFT.  x: (n, ..., 6) -> complex, peak."""
    k = np.arange(Nw)
    bc = (2.0 / Nw) * np.cos(2 * np.pi * harmonic * k / Nw)
    bs = (2.0 / Nw) * np.sin(2 * np.pi * harmonic * k / Nw)
    xr = signal.lfilter(bc, [1.0], x, axis=0)
    xi = signal.lfilter(bs, [1.0], x, axis=0)
    return xr + 1j * xi


def moving_sum(x, Nw, axis=0):
    return signal.lfilter(np.ones(Nw), [1.0], x, axis=axis)


# ------------------------------------------------------------------- relay
class Relay:
    def __init__(self, table=None, settings=None, theta_err=0.0):
        self.t = table or sc.TABLE_I
        self.s = dict(SET); self.s.update(settings or {})
        b = sc.bases(self.t)
        self.I_base, self.V_base = b["I_base"], b["Z_base"] * 0 + b["V_base"]
        self.E_base = self.s["Nw"] * b["I_base"] ** 2
        lc = sc.line_constants(self.t)
        md = sc.modal_transform(lc, self.t)
        S = sc.S6()
        w = lc["w"]
        # The relay's assumed line constants theta are those of the line the
        # EMT model actually represents: the real constant-parameter modal
        # representation.  Z = Tv diag(z_m) Ti^-1,  Y = Ti diag(y_m) Tv^-1.
        zm = md["r"] + 1j * w * md["l"]
        ym = 1j * w * md["c"]
        self.Z_km = md["Tv"] @ np.diag(zm) @ np.linalg.inv(md["Ti"])
        self.Y_km = md["Ti"] @ np.diag(ym) @ md["Tv_inv"]
        self.Z_km_exact = lc["Z"]
        self.modal_Z_err = (np.abs(self.Z_km - lc["Z"]).max()
                            / np.abs(lc["Z"]).max())
        Zt = self.t["length_km"] * self.Z_km
        self.ZmL_true = np.diag(S @ Zt @ np.conj(S).T)             # (6,) complex
        self.S6 = S
        z1 = self.ZmL_true[1]
        self.tau_mimic = (z1.imag / z1.real) / (2 * np.pi * self.t["f0"])

    # ------------------------------------------------------------- pipeline
    def process(self, v, i, dt, t_fault, theta_err=None, ideal_i=None):
        """
        v, i : (n, B, 6) relay-band samples at fs
        Returns a dict of per-case relay outputs and the feature tensor.
        """
        s, Nw = self.s, int(self.s["Nw"])
        n, B, _ = v.shape
        S6 = self.S6
        V = dft_phasors(v, Nw)
        if s.get("mimic", True):
            # d.c.-offset removal (mimic) filter, tuned to the positive-sequence
            # X/R of the protected line.  AMENDMENT to Algorithm 1 line 1: a
            # full-cycle cosine-sine DFT does not reject the exponentially
            # decaying d.c. component, which biases the reactance measurement.
            # The filter is applied only to the fundamental current phasor; the
            # harmonic, transient and CT-health features use the unfiltered
            # current.  The complex fundamental gain is divided out, so both the
            # magnitude and the 81 deg phase advance of the filter are removed.
            im_, g = mimic_filter(i, 1.0 / s["fs"], self.tau_mimic)
            I = dft_phasors(im_, Nw) / g
        else:
            I = dft_phasors(i, Nw)
        Vs = np.einsum('kj,nbj->nbk', S6, V)
        Is = np.einsum('kj,nbj->nbk', S6, I)
        mem = int(s["mem_cycles"] * Nw)

        def superimposed(k0=None):
            """Superimposed sequence quantities against the pre-fault reference.

            Algorithm 1 line 4 specifies a PRE-FAULT reference, not a sliding
            memory: once a disturbance has been detected the reference must be
            frozen at its last pre-disturbance value, otherwise the memory
            catches up with the fault and the superimposed quantities decay to
            zero two cycles after inception.  k0 is the per-case freeze index;
            before it the reference slides, after it it is held.
            """
            kk = np.arange(nT)[:, None]
            idx = kk - mem
            if k0 is not None:
                idx = np.where(kk >= k0[None, :], (k0 - mem)[None, :], idx)
            idx = np.clip(idx, 0, nT - 1)
            g = np.broadcast_to(idx[:, :, None], (nT, B, 6))
            # A sliding full-cycle DFT returns a phasor that advances by
            # 2*pi/Nw per sample, so a reference held at a fixed index must be
            # carried forward at nominal frequency before it is subtracted.
            # For the sliding branch the lag is an integer number of cycles and
            # the rotation is unity, so the same expression covers both.
            rot = np.exp(2j * np.pi * (kk - idx) / Nw)[:, :, None]
            dV = Vs - np.take_along_axis(Vs, g, axis=0) * rot
            dI = Is - np.take_along_axis(Is, g, axis=0) * rot
            # neither the phasors nor the memory are valid until a full window
            # and the memory span have elapsed
            dV[:valid] = 0; dI[:valid] = 0
            return dV, dI

        nT = n
        valid = mem + Nw
        dVs, dIs = superimposed()

        # ---- energies (13), ratios (14), phase differences
        EkI = moving_sum(np.abs(Is) ** 2, Nw)
        EkV = moving_sum(np.abs(Vs) ** 2, Nw)
        eps = 1e-3 * self.I_base
        aI = np.abs(Is)
        Rkl = aI[..., :, None] / (aI[..., None, :] + eps)          # (n,B,6,6)
        iu = np.triu_indices(6, 1); il = np.tril_indices(6, -1)
        Rkl_f = np.concatenate([Rkl[..., iu[0], iu[1]], Rkl[..., il[0], il[1]]], -1)
        th = np.angle(Is)
        Dth = np.angle(np.exp(1j * (th[..., :, None] - th[..., None, :])))
        Dth_f = Dth[..., iu[0], iu[1]]

        # ---- harmonics and transient energy
        I2 = dft_phasors(i, Nw, 2); I3 = dft_phasors(i, Nw, 3)
        fund = np.maximum(np.abs(I).max(-1), 0.05 * self.I_base)
        H2 = np.abs(I2).max(-1) / fund
        H3 = np.abs(I3).max(-1) / fund
        THD = np.zeros_like(H2)
        for h in (2, 3, 5, 7):
            THD = THD + (np.abs(dft_phasors(i, Nw, h)).max(-1) / fund) ** 2
        THD = np.sqrt(THD)
        det = signal.lfilter(DB4_H, [1.0], i, axis=0)
        Etr = moving_sum((det ** 2).sum(-1), Nw)
        eta_tr = Etr / self.E_base

        # ---- CT health
        d2 = np.diff(i, n=2, axis=0, prepend=i[:1], append=i[-1:])[:n]
        pk = signal.lfilter(np.ones(Nw), [1.0], np.abs(d2), axis=0)
        mx = np.zeros_like(pk)
        for c in range(6):
            mx[..., c] = signal.order_filter(np.abs(d2[..., c]), np.ones(Nw), Nw - 1) \
                if False else moving_max(np.abs(d2[..., c]), Nw)
        rms = np.sqrt(moving_sum(d2 ** 2, Nw) / Nw) + 1e-12
        sharp = (mx / rms).max(-1)
        q_CT = np.clip((sharp - s["s0_CT"]) / (s["s1_CT"] - s["s0_CT"]), 0.0, 1.0)
        eta_CT = 1.0 - q_CT

        # ---- noise estimate
        hp = signal.lfilter(DB4_H, [1.0], i, axis=0)
        sig_eta = np.sqrt(moving_sum(hp ** 2, Nw) / Nw).max(-1) / self.I_base

        # ---- disturbance index (15) and adaptive threshold (16)
        Psi = (s["wI"] * (np.abs(dIs) ** 2).sum(-1) / self.I_base ** 2
               + s["wV"] * (np.abs(dVs) ** 2).sum(-1) / self.V_base ** 2
               + s["wT"] * eta_tr)
        kf = int(round(t_fault / dt))
        pre = slice(max(Nw + mem, kf - 2 * Nw), max(Nw + mem + 1, kf - Nw))
        sig_Psi = Psi[pre].std(axis=0)
        # fault-resistance pre-estimate from the superimposed positive sequence
        z1 = self.ZmL_true[1]                       # k = 1: positive sequence
        Zh = dVs[..., 1] / (dIs[..., 1] + 1e-9)
        d_ra = np.clip(Zh.imag / z1.imag, 0.0, 1.5) * self.t["length_km"]
        Rf_hat = np.clip(Zh.real - (d_ra / self.t["length_km"]) * z1.real, 0.0, 5000.0)
        tau = (s["tau0"] + s["kL"] * sig_Psi[None, :] + s["kC"] * q_CT
               + s["kN"] * sig_eta - s["kR"] * np.minimum(Rf_hat / s["Rf_max"], 1.0))
        tau = np.maximum(tau, s["tau0"] - s["kR"])
        D = Psi > tau
        Dp = persist(D, int(s["m_pers"]))
        # freeze the superimposed reference at the first sustained crossing and
        # recompute every quantity that depends on it
        D[:valid] = False
        Dp = persist(D, int(s["m_pers"]))
        any_ = Dp.any(axis=0)
        k0 = np.where(any_, np.argmax(Dp, axis=0), nT + 1)
        if any_.any():
            dVs, dIs = superimposed(k0)
            Psi = (s["wI"] * (np.abs(dIs) ** 2).sum(-1) / self.I_base ** 2
                   + s["wV"] * (np.abs(dVs) ** 2).sum(-1) / self.V_base ** 2
                   + s["wT"] * eta_tr)
            D = Psi > tau
            D[:valid] = False
            Dp = persist(D, int(s["m_pers"]))

        # ---- HIF index (18)-(19) on the measured residual current
        ires = i.sum(-1)
        kap = half_cycle_asymmetry(ires, Nw)
        res_rms = np.sqrt(moving_sum(ires ** 2, Nw) / Nw)
        kap = np.where(res_rms > 0.02 * self.I_base, kap, 0.0)
        # Eq. (18) exactly as printed: the zero- and negative-sequence TOTAL
        # currents under the index convention of Section 2.3 (k = 0 zero,
        # k = 5 negative).  This is only usable with the transform handedness
        # declared there; under the conjugate transform of the shipped
        # reproducibility script the k = 5 entry carries the whole positive-
        # sequence load current and Gamma_HIF exceeds Gamma_th with no fault
        # present.  See spconst.S6.
        Gam = (s["a1"] * (np.abs(Is[..., 0]) + np.abs(Is[..., 5])) / self.I_base
               + s["a2"] * eta_tr + s["a3"] * THD + s["a4"] * kap)
        return dict(V=V, I=I, Vs=Vs, Is=Is, dVs=dVs, dIs=dIs, EkI=EkI, EkV=EkV,
                    Rkl=Rkl_f, Dth=Dth_f, H2=H2, H3=H3, THD=THD, eta_tr=eta_tr,
                    eta_CT=eta_CT, q_CT=q_CT, sig_eta=sig_eta, Psi=Psi, tau=tau,
                    D=Dp, Gam=Gam, kappa=kap, Rf_hat=Rf_hat, sig_Psi=sig_Psi,
                    Nw=Nw, mem=mem, dt=dt, t_fault=t_fault)

    # ---------------------------------------------------------- feature set
    def features(self, o, k):
        """The 85-dimensional feature vector of (11) at window index k."""
        Is, Vs = o["Is"][k], o["Vs"][k]
        f = [np.abs(Is), np.angle(Is), np.abs(Vs), np.angle(Vs),
             o["EkI"][k], o["EkV"][k], o["Rkl"][k], o["Dth"][k],
             o["H2"][k][:, None], o["H3"][k][:, None],
             o["eta_tr"][k][:, None], o["eta_CT"][k][:, None]]
        return np.concatenate(f, axis=-1)

    # --------------------------------------------------------------- locator
    def locate(self, o, k, mask, grounded, theta_scale=None, sel=None,
               d_shunt=None, _iter=1):
        """
        Mask-conditioned robust locator, Eq. (23)-(24).

        The admissible set M(b_f) is the set of fault LOOPS selected by the
        classified mask: one ground loop per faulted conductor when the fault
        involves earth, one phase loop per faulted conductor pair when it does
        not.  Each loop l contributes the exact single-ended relation

            A_l = (d/L) B_l + R_f C_l ,        A_l, B_l, C_l complex,

        with A_l the loop voltage, B_l the total-line mutual-compensated loop
        current impedance drop, and C_l the superimposed loop current, which is
        the polarising quantity of [24].  d and R_f are estimated jointly by a
        Huber M-estimator on the modulus of the complex residual, over a grid
        in d with R_f eliminated in closed form at each d.  The dispersion of
        the per-loop reactance-method estimates is the cross-mode dispersion
        r_d of the security gate (21).
        """
        s = self.s
        L = self.t["length_km"]
        if sel is None:
            V, I = o["V"][k], o["I"][k]
            Ip = o["I"][k - o["mem"]]
        else:
            V, I = o["V"][k][sel], o["I"][k][sel]
            Ip = o["I"][k - o["mem"]][sel]
        dI = I - Ip
        B = V.shape[0]
        if d_shunt is None:
            d_shunt = np.full(B, 0.5 * L)
        Zkm = self.Z_km if theta_scale is None else self.Z_km * theta_scale[:, None, None]
        Zk3 = np.broadcast_to(Zkm, (B, 6, 6)) if Zkm.ndim == 2 else Zkm
        nl = 12
        A = np.zeros((B, nl), complex); Bc = np.zeros((B, nl), complex)
        C = np.zeros((B, nl), complex); act = np.zeros((B, nl), bool)
        dIres = dI.sum(1)
        Ires = I.sum(1)
        z1L, z0L = self.ZmL_true[1], self.ZmL_true[0]
        k0 = (z0L - z1L) / (6.0 * z1L)
        # shunt-charging (nominal-pi) correction: half the charging current of
        # the faulted section is subtracted from the measured current before the
        # loop impedance drop is formed.  d_shunt is refined once.
        Ic = I - 0.5 * d_shunt[:, None] * np.einsum('pq,bq->bp', self.Y_km, V)
        ZI = np.einsum('bpq,bq->bp', Zk3, Ic)
        for p in range(6):
            gnd = grounded & mask[:, p]
            # loop (a): full-matrix mutual compensation
            A[:, p] = np.where(gnd, V[:, p], 0)
            Bc[:, p] = np.where(gnd, ZI[:, p] * L, 0)
            C[:, p] = np.where(gnd, dIres, 0)
            act[:, p] = gnd
            # loop (b): classical residual compensation with k0
            A[:, 6 + p] = np.where(gnd, V[:, p], 0)
            Bc[:, 6 + p] = np.where(gnd, z1L * (I[:, p] + k0 * Ires), 0)
            C[:, 6 + p] = np.where(gnd, dIres, 0)
            act[:, 6 + p] = gnd
        # phase loops for ungrounded faults: consecutive faulted pairs
        ung = (~grounded) & (mask.sum(1) >= 2)
        if ung.any():
            for b in np.nonzero(ung)[0]:
                ps = np.nonzero(mask[b])[0]
                for j in range(len(ps) - 1):
                    p, q = ps[j], ps[j + 1]
                    A[b, j] = V[b, p] - V[b, q]
                    Bc[b, j] = (ZI[b, p] - ZI[b, q]) * L
                    C[b, j] = 0.5 * (dI[b, p] - dI[b, q])
                    act[b, j] = True
        # per-loop reactance-method estimate (Eq. 20 projection form)
        num = np.imag(A * np.conj(C)); den = np.imag(Bc * np.conj(C))
        dl = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), np.nan) * L
        dl = np.where(act, dl, np.nan)
        allnan = ~np.isfinite(dl).any(axis=1)
        dl_s = np.where(np.isfinite(dl), dl, np.nan)
        dl_s[allnan, 0] = 1e9
        with np.errstate(invalid='ignore'):
            d_med = np.nanmedian(dl_s, axis=1)
            r_d = (np.nanmax(dl_s, 1) - np.nanmin(dl_s, 1)) / L
        r_d = np.where(np.isfinite(r_d), r_d, 0.0)
        # joint Huber fit over a grid in d, R_f eliminated at each d
        grid = np.linspace(-0.10 * L, 1.10 * L, 481)
        w = 1.0 / (np.abs(Bc) ** 2 + 1e-9)
        res_best = None
        Ag = A[:, None, :] - (grid[None, :, None] / L) * Bc[:, None, :]
        Cg = np.broadcast_to(C[:, None, :], Ag.shape)
        num_r = (np.real(Ag * np.conj(Cg)) * act[:, None, :]).sum(-1)
        den_r = ((np.abs(Cg) ** 2) * act[:, None, :]).sum(-1) + 1e-12
        Rf_g = np.clip(num_r / den_r, 0.0, s["Rf_max"] * 5)
        resid = np.abs(Ag - Rf_g[:, :, None] * Cg)
        aa = np.abs(np.where(act, A, np.nan))
        aa[allnan, 0] = 1.0
        with np.errstate(invalid='ignore'):
            sc_ = np.nanmedian(aa, axis=1)[:, None, None] + 1e-9
        r = resid / sc_
        dH = s["delta_H"]
        rho = np.where(r <= dH, 0.5 * r ** 2, dH * (r - 0.5 * dH))
        J = (rho * (w * act)[:, None, :]).sum(-1)
        J = np.where(np.isfinite(J), J, np.inf)
        jb = np.argmin(J, axis=1)
        d_hat = np.clip(grid[jb], 0.0, L)
        Rf_hat = Rf_g[np.arange(B), jb]
        if _iter > 0:
            return self.locate(o, k, mask, grounded, theta_scale=theta_scale, sel=sel,
                               d_shunt=np.clip(d_hat, 0.0, L), _iter=_iter - 1)
        return dict(d_hat=d_hat, d_loop=dl, r_d=r_d, d_med=d_med,
                    Rf_hat=Rf_hat, active=act)

    def gate(self, loc, Cf=None):
        s = self.s; L = self.t["length_km"]
        dm = np.where(np.isfinite(loc["d_med"]), loc["d_med"], 1e9)
        g1 = (dm >= -s["beta_rev"] * L) & (dm <= s["beta"] * L)
        g2 = loc["r_d"] <= s["r_max"]
        g3 = np.ones_like(g1) if Cf is None else (Cf >= s["C_min"])
        return g1 & g2 & g3

# ------------------------------------------------------------------ helpers
def mimic_filter(x, dt, tau, f0=50.0):
    """y[n] = (tau/dt)(x[n]-x[n-1]) + x[n].  Returns y and the complex
    fundamental gain g, by which the resulting phasor must be divided."""
    k = tau / dt
    y = signal.lfilter([1.0 + k, -k], [1.0], x, axis=0)
    g = (1.0 + k) - k * np.exp(-1j * 2 * np.pi * f0 * dt)
    return y, g


def persist(D, m):
    """True where D has been true for m consecutive samples."""
    c = moving_sum(D.astype(np.float64), m)
    return c >= m - 1e-9


def moving_max(x, Nw):
    n = x.shape[0]
    out = np.empty_like(x)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.repeat(x[:1], Nw - 1, axis=0), x], axis=0)
    sw = sliding_window_view(pad, Nw, axis=0)
    return sw.max(-1)


def half_cycle_asymmetry(x, Nw):
    """kappa_i of Eq. (19), evaluated on a sliding one-cycle window of the
    measured residual current."""
    p = moving_sum(np.maximum(x, 0.0), Nw)
    m = moving_sum(np.maximum(-x, 0.0), Nw)
    return np.abs(p - m) / (p + m + 1e-12)
