#!/usr/bin/env python3
"""
spmeas.py -- relay measurement chain: current transformer, capacitor voltage
transformer, additive noise, anti-aliasing filter and decimation.

Declared parameters are those of Table I.  Three quantities that Table I does
not declare are added here and are flagged as ADDED in the manuscript:
the CT secondary winding resistance, the unsaturated magnetising inductance,
and the air-core (saturated) inductance.
"""
import numpy as np
from scipy import signal

# ---- CT, class 5P20, 200:1, knee 150 V (Table I) --------------------------
CT = dict(ratio=200.0, knee_V=150.0, R_sec=0.20, burden_rated=0.5,
          burden_2x=1.0, Lm_H=13.5, Lair_H=0.0135, I_sec_rated=5.0)
# CVT: series-tuned capacitive divider, unity transfer at f0 by construction
# (omega0^2 L C = 1), quality factor Q setting the subsidence-transient decay.
CVT = dict(ratio=845.0, Q=4.0, f0=50.0, model="ideal")


def ct_secondary(i_prim, dt, burden, remanence_pu, ct=CT):
    """
    Saturable current transformer.

    i_prim      (n, B, 6)  primary current [A]
    burden      (B,)       total burden resistance [ohm]
    remanence   (B,)       initial flux as a fraction of the knee flux

    Returns the secondary current referred to the primary [A], i.e. N*i_sec,
    so that it is directly comparable with i_prim.  Saturation shows up as
    a deficit relative to i_prim.

    Model (IEEE Std C37.110):
        dlambda/dt = R_tot ( i_prim/N - i_e )
        i_e(lambda) = lambda/Lm + sign(lambda) (|lambda|-lambda_k)_+ / Lair
    """
    i_prim = np.asarray(i_prim, np.float64)
    n, B, _ = i_prim.shape
    N = ct["ratio"]
    lam_k = np.sqrt(2.0) * ct["knee_V"] / (2 * np.pi * 50.0)
    Rt = (burden + ct["R_sec"])[:, None]                       # (B,1)
    lam = (remanence_pu[:, None] * lam_k) * np.ones((B, 6))
    per = int(round(1.0 / (50.0 * dt)))
    out = np.empty(i_prim.shape, np.float32)
    invLm, invLa = 1.0 / ct["Lm_H"], 1.0 / ct["Lair_H"]
    for k in range(n):
        ip = i_prim[k] / N
        ie = lam * invLm + np.sign(lam) * np.maximum(np.abs(lam) - lam_k, 0.0) * invLa
        isec = ip - ie
        lam = lam + dt * Rt * isec
        out[k] = isec * N
    return out


def cvt_transfer(v_prim, dt, cvt=CVT, n_pad=None, model=None):
    """
    Capacitor voltage transformer.

    The coupling capacitance is tuned by a series reactor so that the divider
    is exactly unity at the fundamental; the resulting second-order transfer

        H(s) = (w0/Q) s / (s^2 + (w0/Q) s + w0^2)

    reproduces the subsidence transient that follows a voltage collapse, with
    decay time constant 2Q/w0.  Q is a declared parameter.

    The record is pre-padded with a whole number of cycles of its own leading
    (steady-state) segment so that no filter start-up transient reaches the
    pre-fault window.
    """
    if (model or cvt["model"]) == "ideal":
        return v_prim
    w0 = 2 * np.pi * cvt["f0"]; Q = cvt["Q"]
    b, a = signal.bilinear([w0 / Q, 0.0], [1.0, w0 / Q, w0 ** 2], fs=1.0 / dt)
    per = int(round(1.0 / (cvt["f0"] * dt)))
    npad = n_pad if n_pad is not None else 20 * per
    reps = int(np.ceil(npad / per))
    pad = np.tile(v_prim[:per], (reps,) + (1,) * (v_prim.ndim - 1))[-npad:]
    y = signal.lfilter(b, a, np.concatenate([pad, v_prim], axis=0).astype(np.float64), axis=0)
    return y[npad:]


def add_noise(x, snr_db, rng, kind="gaussian", ref_slice=None):
    """Additive measurement noise at the declared SNR (Eq. 26).

    The signal power E_x is referred to the PRE-FAULT window, not to the whole
    record: a 20 dB record-referred SNR on a record containing a 3 kA fault
    would place more noise on the pre-fault window than the load current
    itself, which is not what a measurement signal-to-noise ratio means."""
    xr = x if ref_slice is None else x[ref_slice]
    p = (xr.astype(np.float64) ** 2).mean(axis=0)               # (B,6)
    sig = np.sqrt(p / (10.0 ** (np.asarray(snr_db, float)[:, None] / 10.0)))
    e = rng.standard_normal(x.shape)
    if kind == "coloured":
        e = np.cumsum(e, axis=0); e /= (e.std(axis=0, keepdims=True) + 1e-30)
    elif kind == "impulsive":
        m = rng.random(x.shape) < 0.001
        e = e + m * rng.standard_normal(x.shape) * 12.0
        e /= e.std(axis=0, keepdims=True)
    return x + sig[None] * e


def antialias_decimate(x, dt_in, fc, fs_out, order=4):
    """Butterworth anti-aliasing filter at fc followed by decimation to fs_out."""
    fs_in = 1.0 / dt_in
    sos = signal.butter(order, fc, btype="low", fs=fs_in, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=0)
    step = int(round(fs_in / fs_out))
    return y[::step], step


def quantise(x, n_bits=16, full_scale=None):
    """Uniform mid-tread quantiser, +/- full_scale."""
    fs = full_scale if full_scale is not None else np.abs(x).max(axis=0, keepdims=True) * 1.2
    q = 2.0 * fs / (2 ** n_bits)
    return np.round(x / q) * q
