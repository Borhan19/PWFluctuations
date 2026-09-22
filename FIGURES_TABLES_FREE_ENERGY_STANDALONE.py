#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRL figures and Supplemental-Material tables for
"Physical-Work Fluctuation Relations from Accessible Quantum Macrostates".

This is a standalone, reader-facing plotting/table script.  It does two things:

  (1) Recomputes the nominal L=Q=4 Bose-Hubbard protocol from the Hamiltonian
      (initial Gibbs state, unitary ramp, endpoint MaxEnt fit, Target-B terms,
      work/record trajectory distribution).

  (2) Reads the already validated sweep/calibration CSV files in ./input_data
      for the expensive information-sampling and robustness campaigns.

Run by clicking Run in Spyder/Jupyter/VS Code, or from a terminal:
    python PRL_FIGURES_TABLES_STANDALONE.py

Outputs are written to:
    ./PRL_figures_tables_final/

Required Python packages:
    numpy, scipy, pandas, matplotlib

No project-specific module is required.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, polar
from scipy.optimize import least_squares, linprog, minimize, minimize_scalar
from scipy.special import logsumexp


# =============================================================================
# 0. USER SETTINGS
# =============================================================================

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = HERE / "input_data"
OUT_DIR = HERE / "PRL_figures_tables_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Core nominal time grid.  0.1/J resolution includes the known 15.2/J and 17.3/J
# structure while remaining fast enough for a one-click standalone script.
TIME_STEP = 0.2

# Publication output.
PNG_DPI = 300
SAVE_PNG = True
SAVE_PDF = True

# Show every generated figure in Jupyter/Spyder as well as saving it.
# Set False for a completely silent batch run.
SHOW_FIGURES = True

# Figure geometry. The 7.05-inch figures are designed for a two-column PRL
# figure* environment. If they are inserted into a single column and scaled
# down with \linewidth, every font is also scaled down and will look too small.
PRL_DOUBLE_COLUMN_WIDTH = 7.05
PRL_SINGLE_COLUMN_WIDTH = 3.35

# Readable publication typography at the FINAL printed size.
FONT_BASE = 9.5
FONT_LABEL = 9.5
FONT_TICK = 8.6
FONT_LEGEND = 8.0
FONT_PANEL = 10.5
FONT_SMALL = 7.8

# High-contrast, perceptually uniform heat-map palette.
HEATMAP_CMAP = "magma"

# Sampling benchmark used throughout the project.
REL_TOL = 0.01       # 1% error in the exponential average
CONFIDENCE = 0.99    # 99% confidence
ETA = 1.0 - CONFIDENCE

# If True, recompute the variance-optimal reference from the nominal trajectory
# ensemble.  This is inexpensive compared with the large robustness sweeps.
RECOMPUTE_VARIANCE_OPTIMUM = True

# Fast publication mode: use the frozen validated nominal data shipped in
# input_data. Set False only if you want to rebuild the nominal dynamics.
USE_PRECOMPUTED_CORE = True


# =============================================================================
# 1. PLOT STYLE
# =============================================================================

def set_prl_style():
    """Readable PRL-style typography with a safe Times fallback."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.linewidth": 0.85,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.65,
        "ytick.minor.width": 0.65,
        "xtick.major.size": 3.4,
        "ytick.major.size": 3.4,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.2,
        "legend.frameon": False,
        "figure.constrained_layout.use": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label, x=0.02, y=0.98):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_PANEL, fontweight="bold")


def save_figure(fig, stem):
    """Save a figure and, by default, also display it in the notebook."""
    if SAVE_PDF:
        fig.savefig(OUT_DIR / f"{stem}.pdf")
    if SAVE_PNG:
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=PNG_DPI)

    if SHOW_FIGURES:
        # plt.show() renders inline in Jupyter and in Spyder's Plots pane.
        # Closing afterwards releases memory without removing the rendered output.
        plt.show()

    plt.close(fig)


# =============================================================================
# 2. NOMINAL BOSE-HUBBARD MODEL
# =============================================================================

L = 4
Q = 4
J = 1.0
UQ = 1.3 / 3.0
LAMBDA0 = 20.0
TAU = 20.0
BETA_I = 0.2

BASIS = [occ for occ in itertools.product(range(Q + 1), repeat=L)
         if sum(occ) == Q]
DIM = len(BASIS)
INDEX = {b: i for i, b in enumerate(BASIS)}


def bh_hamiltonian(Lam: float) -> np.ndarray:
    """Fixed-Q Bose-Hubbard Hamiltonian in units J=hbar=kB=1."""
    H = np.zeros((DIM, DIM), dtype=complex)

    for i, b in enumerate(BASIS):
        n = np.asarray(b, dtype=float)
        H[i, i] += UQ / 2.0 * np.sum(n * (n - 1.0))
        H[i, i] += Lam * np.sum(n[L // 2:])

    for i, b0 in enumerate(BASIS):
        b = list(b0)
        for j in range(L - 1):
            # j+1 -> j
            if b[j + 1] > 0:
                nb = b.copy()
                amp = math.sqrt((nb[j] + 1) * nb[j + 1])
                nb[j] += 1
                nb[j + 1] -= 1
                H[INDEX[tuple(nb)], i] += -J * amp
            # j -> j+1
            if b[j] > 0:
                nb = b.copy()
                amp = math.sqrt((nb[j + 1] + 1) * nb[j])
                nb[j + 1] += 1
                nb[j] -= 1
                H[INDEX[tuple(nb)], i] += -J * amp

    return (H + H.conj().T) / 2.0


P_R = []
for r in range(Q + 1):
    diagonal = np.array([1.0 if sum(b[L // 2:]) == r else 0.0 for b in BASIS])
    P_R.append(np.diag(diagonal))

N_R = sum(r * P_R[r] for r in range(Q + 1))
H_I = bh_hamiltonian(LAMBDA0)
H_F = bh_hamiltonian(0.0)
E_I, V_I = eigh(H_I)
E_F, V_F = eigh(H_F)


def Lambda_of_t(t):
    return LAMBDA0 * (1.0 - t / TAU)


def canonical_state(H, beta):
    e, V = eigh(H)
    x = -beta * e
    logZ = logsumexp(x)
    p = np.exp(x - logZ)
    rho = (V * p) @ V.conj().T
    return rho, e, V, p, float(logZ)


RHO_I, _, _, Q_I, LOGZ_I = canonical_state(H_I, BETA_I)
S_I = -float(np.sum(Q_I * np.log(Q_I)))
E_B = float(np.real(np.trace(H_I @ RHO_I)))
P_B = np.array([np.real(np.trace(P @ RHO_I)) for P in P_R])


def propagate_nominal_dense():
    """High-accuracy DOP853 propagator with dense output."""
    def rhs(t, y):
        U = y.reshape((DIM, DIM))
        H = H_F + Lambda_of_t(t) * N_R
        return (-1j * H @ U).reshape(-1)

    y0 = np.eye(DIM, dtype=complex).reshape(-1)
    sol = solve_ivp(rhs, (0.0, TAU), y0, method="DOP853",
                    rtol=1e-11, atol=1e-13, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def stable_gibbs_from_generator(K):
    """Return exp(-K)/Z and log Z without overflow."""
    vals, vecs = eigh((K + K.conj().T) / 2.0)
    x = -vals
    logZ = logsumexp(x)
    weights = np.exp(x - logZ)
    rho = (vecs * weights) @ vecs.conj().T
    return rho, float(logZ)


def maxent_constraints(theta, H):
    """Gauge choice alpha_4=0. theta=(beta,alpha0,...,alpha3)."""
    beta = float(theta[0])
    alpha = np.r_[np.asarray(theta[1:], dtype=float), 0.0]
    A = sum(alpha[r] * P_R[r] for r in range(Q + 1))
    rho_bar, logZ = stable_gibbs_from_generator(beta * H + A)
    E = float(np.real(np.trace(H @ rho_bar)))
    p = np.array([np.real(np.trace(P @ rho_bar)) for P in P_R])
    return np.r_[E, p[:4]], rho_bar, logZ, alpha


def fit_maxent(H, E_target, p_target, x0):
    target = np.r_[E_target, p_target[:4]]

    def residual(x):
        vals, _, _, _ = maxent_constraints(x, H)
        return vals - target

    res = least_squares(residual, x0, xtol=2e-12, ftol=2e-12, gtol=2e-12,
                        max_nfev=2500, x_scale="jac")
    vals, rho_bar, logZ, alpha = maxent_constraints(res.x, H)
    max_resid = float(np.max(np.abs(vals - target)))
    if max_resid > 2e-8:
        warnings.warn(f"MaxEnt residual {max_resid:.3e} is larger than expected.")
    return res.x, rho_bar, logZ, alpha, max_resid


def dephase_in_energy(rho, H):
    e, V = eigh(H)
    diag = np.real(np.diag(V.conj().T @ rho @ V))
    rho_d = (V * diag) @ V.conj().T
    return rho_d, e, V, diag


def endpoint_quantities(rho, H, theta, logZ, alpha):
    beta = float(theta[0])
    E = float(np.real(np.trace(H @ rho)))
    p = np.array([np.real(np.trace(P @ rho)) for P in P_R])

    rho_d, e, V, diag = dephase_in_energy(rho, H)
    pseq = np.array([np.real(np.trace(P @ rho_d)) for P in P_R])

    # Sequential normalization Z_seq = Tr[e^{-beta H} e^{-A}].
    x = -beta * e
    # Keep the absolute normalization here, since energies are moderate.
    exp_beta_H = (V * np.exp(x)) @ V.conj().T
    exp_minus_A = sum(math.exp(-alpha[r]) * P_R[r] for r in range(Q + 1))
    Zseq = float(np.real(np.trace(exp_beta_H @ exp_minus_A)))
    Qnc = math.log(Zseq) - logZ

    Delta_meas = float(np.dot(alpha, pseq - p))
    Drel = -S_I + beta * E + float(np.dot(alpha, p)) + logZ
    mean_sigma = Drel + Delta_meas + Qnc

    return {
        "E": E,
        "p": p,
        "pseq": pseq,
        "D": Drel,
        "Delta_meas": Delta_meas,
        "Qnc": Qnc,
        "mean_sigma": mean_sigma,
        "Zseq": Zseq,
        "beta": beta,
        "alpha": alpha,
    }


def compute_time_resolved_nominal(sol):
    """Recompute the thermodynamic endpoint at every truncated protocol time."""
    times = np.unique(np.r_[np.arange(0.0, TAU + 0.5 * TIME_STEP, TIME_STEP), 15.2, 17.3])
    rows = []
    xfit = np.array([BETA_I, 0.0, 0.0, 0.0, 0.0])

    for t in times:
        U = sol.sol(t).reshape((DIM, DIM))
        # Numerical symmetrization/normalization is enough at the selected tolerances.
        rho = U @ RHO_I @ U.conj().T
        rho = (rho + rho.conj().T) / 2.0
        rho /= np.real(np.trace(rho))
        Ht = H_F + Lambda_of_t(t) * N_R
        E = float(np.real(np.trace(Ht @ rho)))
        p = np.array([np.real(np.trace(P @ rho)) for P in P_R])
        xfit, _, logZ, alpha, resid = fit_maxent(Ht, E, p, xfit)
        q = endpoint_quantities(rho, Ht, xfit, logZ, alpha)
        row = {
            "Jt": t,
            "Lambda": Lambda_of_t(t),
            "E": q["E"],
            "D": q["D"],
            "Delta_meas": q["Delta_meas"],
            "Qnc": q["Qnc"],
            "mean_sigma": q["mean_sigma"],
            "beta_star": q["beta"],
            "maxent_resid": resid,
        }
        for r in range(Q + 1):
            row[f"p{r}"] = q["p"][r]
            row[f"pseq{r}"] = q["pseq"][r]
            row[f"alpha{r}"] = alpha[r]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "nominal_time_resolved_data.csv", index=False)
    return df


def endpoint_trajectory_ensemble(U_endpoint, endpoint):
    """Positive sequential trajectory ensemble (n,m,r) for canonical Target B."""
    Uem = V_F.conj().T @ U_endpoint @ V_I
    Tmn = np.abs(Uem) ** 2
    Rmr = np.array([
        [np.real(V_F[:, m].conj().T @ P @ V_F[:, m]) for P in P_R]
        for m in range(DIM)
    ])

    beta = endpoint["beta"]
    alpha = endpoint["alpha"]
    log_norm = math.log(endpoint["Zseq"] / math.exp(LOGZ_I))

    records = []
    for n in range(DIM):
        for m in range(DIM):
            base = Q_I[n] * Tmn[m, n]
            if base < 1e-22:
                continue
            W = E_F[m] - E_I[n]
            for r in range(Q + 1):
                prob = base * Rmr[m, r]
                if prob < 1e-24:
                    continue
                sigma = (BETA_I * W
                         + (beta - BETA_I) * E_F[m]
                         + alpha[r]
                         + log_norm)
                records.append((n, m, r, prob, W, sigma, math.exp(-sigma)))

    df = pd.DataFrame(records, columns=["n", "m", "r", "prob", "W", "sigma", "Y"])
    df.to_csv(OUT_DIR / "endpoint_targetB_trajectory_distribution.csv", index=False)
    return df, Tmn, Rmr


# =============================================================================
# 3. FINITE-CONFIDENCE CHERNOFF COST
# =============================================================================

def log_mgf(t, y, p):
    return float(logsumexp(np.log(p) + t * y))


def cramer_rate(y, p, a, upper):
    """Cramer rate for P(mean Y >= a) or P(mean Y <= a)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()

    # The derivative at t=0 is E[Y]-a = 1-a.  For a=1+eps the
    # upper-tail maximizer is positive; for a=1-eps the lower-tail
    # maximizer is negative.
    if upper:
        # Rate maximizers are extremely close to zero for these heavy tails.
        # Optimize t=exp(s) so several decades are handled safely.
        def obj(s):
            t = math.exp(s)
            return -(t * a - log_mgf(t, y, p))
        res = minimize_scalar(obj, bounds=(-30.0, -1.0), method="bounded",
                              options={"xatol": 1e-12})
        return max(0.0, -float(res.fun))
    else:
        def obj(s):
            t = -math.exp(s)
            return -(t * a - log_mgf(t, y, p))
        res = minimize_scalar(obj, bounds=(-30.0, 4.0), method="bounded",
                              options={"xatol": 1e-12})
        return max(0.0, -float(res.fun))


def chernoff_sample_size(y, p, eps=REL_TOL, eta=ETA):
    rp = cramer_rate(y, p, 1.0 + eps, upper=True)
    rm = cramer_rate(y, p, 1.0 - eps, upper=False)
    rate = min(rp, rm)
    if rate <= 0:
        return np.inf, rp, rm
    return math.log(2.0 / eta) / rate, rp, rm


def family_Y_for_theta(theta, traj_df, Rmr):
    """Y=e^{-sigma_theta} for the exact Target-B family; alpha4=0."""
    beta = float(theta[0])
    alpha = np.r_[np.asarray(theta[1:], dtype=float), 0.0]

    # Zseq(theta)=sum_m exp(-beta E_m) sum_r exp(-alpha_r) R_mr.
    record_weight_m = Rmr @ np.exp(-alpha)
    Zseq = float(np.sum(np.exp(-beta * E_F) * record_weight_m))
    log_norm = math.log(Zseq) - LOGZ_I

    W = traj_df["W"].to_numpy()
    m = traj_df["m"].to_numpy(dtype=int)
    r = traj_df["r"].to_numpy(dtype=int)
    sigma = BETA_I * W + (beta - BETA_I) * E_F[m] + alpha[r] + log_norm
    return np.exp(-sigma), sigma


def variance_optimal_theta(traj_df, Rmr, x0):
    p = traj_df["prob"].to_numpy()

    def objective(theta):
        y, _ = family_Y_for_theta(theta, traj_df, Rmr)
        return float(np.dot(p, y * y) - 1.0)

    # L-BFGS-B from several sensible starts.  Wide but finite bounds prevent
    # excursions into numerically irrelevant references.
    starts = [
        np.asarray(x0, dtype=float),
        np.array([BETA_I, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.02, 0.04, 0.02]),
    ]
    bounds = [(-1.0, 5.0)] + [(-5.0, 5.0)] * 4
    best = None
    for s in starts:
        res = minimize(objective, s, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)


# =============================================================================
# 4. LOAD VALIDATED SWEEP DATA
# =============================================================================

def read_required_csv(name):
    candidates = [DATA_DIR / name, HERE / name, Path("/mnt/data") / name]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find {name}. Put it in {DATA_DIR} next to this script."
    )


def load_sweep_data():
    return {
        "frontier": read_required_csv("targetb_nominal_information_sampling_frontier.csv"),
        "frontier_verified": read_required_csv("targetb_nominal_information_sampling_frontier_verified.csv"),
        "full_grid": read_required_csv("targetb_full_5x5x5_time_audit.csv"),
        "endpoint_grid": read_required_csv("targetb_endpoint_multistart_magnus.csv"),
        "arb_fullrank": read_required_csv("arbitrary_initial_state_fullrank_validation.csv"),
        "arb_closure": read_required_csv("arbitrary_initial_state_closure_and_coherence_tests.csv"),
        "arb_support": read_required_csv("arbitrary_initial_state_support_deficit_validation.csv"),
        "arb_metrology": read_required_csv("arbitrary_initial_state_calibration_metrology.csv"),
        "compatible_detail": read_required_csv("compatible_pair_detail.csv"),
        "compatible_final_record": read_required_csv("compatible_pair_final_record.csv"),
        "nominal_time": read_required_csv("nominal_time_resolved_data.csv"),
        "nominal_traj": read_required_csv("endpoint_targetB_trajectory_distribution.csv"),
        "sampling_costs_verified": read_required_csv("main_sampling_costs_verified.csv"),
    }


# =============================================================================
# 5. MAIN FIGURE 1
# =============================================================================

def make_main_figure_1(time_df, endpoint):
    fig = plt.figure(figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.85, 1.35], wspace=0.38)

    # (a) Protocol schematic.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    panel_label(ax, "(a)")
    # Fixed-size boxes avoid text-driven overlap in the narrow PRL panel.
    boxes = [
        (0.02, 0.44, 0.27, 0.22, "Gibbs $B$\n" + r"$\rho_B$"),
        (0.365, 0.44, 0.27, 0.22, "isolated drive\n" + r"$H_i\to H_f$"),
        (0.71, 0.44, 0.27, 0.22, "endpoint $C^{-}$\n" + r"$E_f,p_r$"),
    ]
    for x, y, w, h, txt in boxes:
        patch = FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                               boxstyle="round,pad=0.018", facecolor="0.96",
                               edgecolor="0.25", linewidth=0.75, clip_on=False)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT_SMALL, linespacing=1.22)
    ax.annotate("", xy=(0.36, 0.55), xytext=(0.295, 0.55), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.annotate("", xy=(0.705, 0.55), xytext=(0.64, 0.55), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(0.50, 0.20,
            r"TPM: $E_n^i\to E_m^f\to r$" + "\n" + r"$W_{mn}=E_m^f-E_n^i$",
            transform=ax.transAxes, ha="center", va="center", fontsize=FONT_SMALL + 0.3)
    ax.text(0.50, 0.84, r"retained macrostate: $\bar\rho_{E,R}$",
            transform=ax.transAxes, ha="center", va="center", fontsize=FONT_SMALL + 0.6)

    # (b) Undisturbed vs sequential record.
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b)")
    rvals = np.arange(Q + 1)
    width = 0.37
    ax.bar(rvals - width / 2, endpoint["p"], width=width, label=r"$p_r$")
    ax.bar(rvals + width / 2, endpoint["pseq"], width=width, label=r"$p_r^{\rm seq}$")
    ax.set_xlabel(r"record $r=n_R$")
    ax.set_ylabel("probability")
    ax.set_xticks(rvals)
    ax.set_ylim(0, 0.34)
    ax.legend(loc="upper right", ncol=1)

    # (c) Time-resolved mean decomposition.
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "(c)")
    ax.plot(time_df["Jt"], time_df["D"], label=r"$D(\rho_t\Vert\bar\rho_t)$")
    ax.plot(time_df["Jt"], time_df["mean_sigma"], ls="--",
            label=r"$\langle\sigma_{\rm phys,R}\rangle$")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel("dimensionless")
    ax.set_xlim(0, TAU)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))

    ins = inset_axes(ax, width="47%", height="48%", loc="center", borderpad=0.8)
    ins.plot(time_df["Jt"], time_df["Delta_meas"], label=r"$\Delta_{\rm meas}$")
    ins.plot(time_df["Jt"], time_df["Qnc"], label=r"$\mathcal{Q}_{\rm nc}$")
    ins.set_xlim(0, TAU)
    ins.tick_params(labelsize=7.2)
    ins.legend(fontsize=7.0, loc="upper left")

    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.19, top=0.96)
    save_figure(fig, "Fig1_PRL_physical_work_record")


# =============================================================================
# 6. MAIN FIGURE 2
# =============================================================================

def make_main_figure_2(frontier_raw, frontier_verified, sampling_costs):
    fig, axes = plt.subplots(1, 2, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.75), gridspec_kw={"wspace": 0.38})

    # (a) Sampling-cost comparison.
    ax = axes[0]
    panel_label(ax, "(a)")
    labels = ["Jarzynski", "thermo.\nTarget B", "variance\nopt.", "finite-conf.\nopt."]
    vals = np.array([
        sampling_costs["jarzynski"],
        sampling_costs["thermodynamic"],
        sampling_costs["variance_opt"],
        sampling_costs["finite_conf_opt"],
    ]) / 1e9
    bars = ax.bar(np.arange(4), vals, width=0.68)
    ax.set_ylabel(r"shots $N$ ($10^9$)")
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylim(0, max(vals) * 1.16)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.045 * max(vals),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.0)

    # (b) Information-sampling frontier.
    ax = axes[1]
    panel_label(ax, "(b)")
    raw = frontier_raw.sort_values("D")
    ver = frontier_verified.sort_values("D_achieved")
    N0 = float(ver.loc[np.argmin(np.abs(ver["D_achieved"])), "N_1pct_99pct"])
    ax.plot(raw["D"], raw["N"] / N0, lw=1.0, alpha=0.85, label="optimized frontier")
    ax.plot(ver["D_achieved"], ver["N_1pct_99pct"] / N0, "o", ms=3.2,
            label="verified points")

    # Local square-root law only where it is genuinely local.
    C_IS = 2.02
    dsmall = np.linspace(0.0, 0.01, 150)
    ax.plot(dsmall, 1.0 - C_IS * np.sqrt(dsmall), ls="--",
            label=r"$1-C_{\rm IS}\sqrt{\delta}$")

    dsat = float(ver["D_achieved"].max())
    nsat = float(ver.loc[ver["D_achieved"].idxmax(), "N_1pct_99pct"] / N0)
    ax.axvline(dsat, lw=0.7, ls=":")
    ax.text(dsat * 0.985, nsat + 0.018, r"$\delta_{\rm sat}$",
            ha="right", va="bottom", fontsize=8.0)
    ax.set_xlabel(r"thermodynamic displacement $\delta$")
    ax.set_ylabel(r"$N^\star(\delta)/N^\star(0)$")
    ax.set_xlim(0, 0.046)
    ax.set_ylim(0.75, 1.02)
    ax.legend(loc="upper right")

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.25, top=0.96)
    save_figure(fig, "Fig2_PRL_information_sampling")


# =============================================================================
# 6B. RARE-EVENT TRAJECTORY HEAT MAPS
# =============================================================================

def trajectory_heatmap_matrices(traj_df):
    """Aggregate trajectory weights over the record label r on an (m,n) grid.

    Returns
    -------
    mean_share : array, shape (DIM, DIM)
        Fractional contribution of each (n,m) energy-sector pair to
        <exp(-sigma)> = 1.
    second_share : array, shape (DIM, DIM)
        Fractional contribution of each (n,m) pair to <exp(-2 sigma)>.
        This is the most direct heat-map view of the rare-event sampling burden.
    top_initial : ndarray
        Two initial energy sectors with the largest column-integrated second-moment
        contribution. These are outlined on the second-moment heat map.
    """
    required = {"n", "m", "prob", "Y"}
    missing = required.difference(traj_df.columns)
    if missing:
        raise ValueError(f"Trajectory dataframe is missing columns: {sorted(missing)}")

    first = np.zeros((DIM, DIM), dtype=float)
    second = np.zeros((DIM, DIM), dtype=float)

    for row in traj_df.itertuples(index=False):
        n = int(row.n)
        m = int(row.m)
        prob = float(row.prob)
        y = float(row.Y)
        first[m, n] += prob * y
        second[m, n] += prob * y * y

    first_total = float(first.sum())
    second_total = float(second.sum())
    if first_total <= 0.0 or second_total <= 0.0:
        raise RuntimeError("Heat-map contribution matrices have non-positive total weight.")

    mean_share = first / first_total
    second_share = second / second_total

    second_by_initial = second_share.sum(axis=0)
    top_initial = np.argsort(second_by_initial)[-2:]

    # Save the plotted numerical data so the heat map is fully reproducible.
    cols = [f"n{n}" for n in range(DIM)]
    idx = [f"m{m}" for m in range(DIM)]
    pd.DataFrame(mean_share, index=idx, columns=cols).to_csv(
        OUT_DIR / "trajectory_heatmap_mean_contribution.csv"
    )
    pd.DataFrame(second_share, index=idx, columns=cols).to_csv(
        OUT_DIR / "trajectory_heatmap_second_moment_contribution.csv"
    )

    return mean_share, second_share, top_initial


def _heatmap_norm(a, decades=8):
    """Stable logarithmic normalization while keeping exact zero cells blank."""
    positive = np.asarray(a)[np.asarray(a) > 0.0]
    if positive.size == 0:
        raise ValueError("Heat map has no positive entries.")
    vmax = float(positive.max())
    vmin = max(float(positive.min()), vmax * 10.0**(-decades))
    return LogNorm(vmin=vmin, vmax=vmax)


def _draw_sector_heatmap(ax, data, cbar_label, panel=None, highlight_initial=None):
    """Draw one high-contrast trajectory-sector heat map."""
    masked = np.ma.masked_less_equal(data, 0.0)
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=HEATMAP_CMAP,
        norm=_heatmap_norm(data),
        rasterized=True,
    )

    if panel is not None:
        panel_label(ax, panel)

    ax.set_xlabel(r"initial energy sector $n$")
    ax.set_ylabel(r"final energy sector $m$")
    ticks = np.arange(0, DIM, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-0.5, DIM - 0.5)
    ax.set_ylim(-0.5, DIM - 0.5)

    if highlight_initial is not None:
        for n in np.sort(np.asarray(highlight_initial, dtype=int)):
            ax.add_patch(
                Rectangle(
                    (n - 0.5, -0.5), 1.0, DIM,
                    facecolor="none", edgecolor="cyan", linewidth=1.15,
                    clip_on=True,
                )
            )

    cbar = ax.figure.colorbar(im, ax=ax, pad=0.025, fraction=0.050)
    cbar.set_label(cbar_label, fontsize=FONT_SMALL + 0.4)
    cbar.ax.tick_params(labelsize=FONT_SMALL - 0.2, width=0.7)
    return im


def make_main_figure_2_with_heatmap(frontier_raw, frontier_verified, sampling_costs, traj_df):
    """Optional three-panel main-text Fig. 2 with the rare-event heat map as panel (c).

    This is designed at full PRL double-column width. Use it in a figure* environment
    with width=\\textwidth so the text is not scaled down.
    """
    _, second_share, top_initial = trajectory_heatmap_matrices(traj_df)

    fig = plt.figure(figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.08, 1.12], wspace=0.43)

    # (a) Sampling-cost comparison.
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "(a)")
    labels = ["Jarzynski", "thermo.\nTarget B", "variance\nopt.", "finite-conf.\nopt."]
    vals = np.array([
        sampling_costs["jarzynski"],
        sampling_costs["thermodynamic"],
        sampling_costs["variance_opt"],
        sampling_costs["finite_conf_opt"],
    ]) / 1e9
    bars = ax.bar(np.arange(4), vals, width=0.68)
    ax.set_ylabel(r"shots $N$ ($10^9$)")
    ax.set_xticks(np.arange(4), labels)
    ax.tick_params(axis="x", labelsize=FONT_SMALL - 0.3)
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height() + 0.04 * max(vals),
            f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SMALL,
        )

    # (b) Information-sampling frontier.
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b)")
    raw = frontier_raw.sort_values("D")
    ver = frontier_verified.sort_values("D_achieved")
    N0 = float(ver.loc[np.argmin(np.abs(ver["D_achieved"])), "N_1pct_99pct"])
    ax.plot(raw["D"], raw["N"] / N0, lw=1.25, label="optimized frontier")
    ax.plot(
        ver["D_achieved"], ver["N_1pct_99pct"] / N0,
        "o", ms=4.0, label="verified points",
    )
    C_IS = 2.02
    dsmall = np.linspace(0.0, 0.01, 150)
    ax.plot(dsmall, 1.0 - C_IS * np.sqrt(dsmall), ls="--",
            label=r"$1-C_{\rm IS}\sqrt{\delta}$")
    dsat = float(ver["D_achieved"].max())
    nsat = float(ver.loc[ver["D_achieved"].idxmax(), "N_1pct_99pct"] / N0)
    ax.axvline(dsat, lw=0.85, ls=":")
    ax.text(dsat * 0.985, nsat + 0.018, r"$\delta_{\rm sat}$",
            ha="right", va="bottom", fontsize=FONT_SMALL)
    ax.set_xlabel(r"thermodynamic displacement $\delta$")
    ax.set_ylabel(r"$N^\star(\delta)/N^\star(0)$")
    ax.set_xlim(0, 0.046)
    ax.set_ylim(0.75, 1.02)
    ax.legend(loc="upper right", fontsize=FONT_SMALL - 0.1)

    # (c) Rare-event origin of the second moment / sampling burden.
    ax = fig.add_subplot(gs[0, 2])
    _draw_sector_heatmap(
        ax,
        second_share,
        r"share of $\langle e^{-2\sigma}\rangle$",
        panel="(c)",
        highlight_initial=top_initial,
    )

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.965)
    save_figure(fig, "Fig2_PRL_information_sampling_with_heatmap")


def make_sm_figure_trajectory_heatmaps(traj_df):
    """Two-panel detailed heat map for the Supplemental Material."""
    mean_share, second_share, top_initial = trajectory_heatmap_matrices(traj_df)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(PRL_DOUBLE_COLUMN_WIDTH, 3.15),
        gridspec_kw={"wspace": 0.42},
    )

    _draw_sector_heatmap(
        axes[0],
        mean_share,
        r"share of $\langle e^{-\sigma}\rangle$",
        panel="(a)",
    )
    _draw_sector_heatmap(
        axes[1],
        second_share,
        r"share of $\langle e^{-2\sigma}\rangle$",
        panel="(b)",
        highlight_initial=top_initial,
    )

    second_by_n = second_share.sum(axis=0)
    top_sorted = np.sort(top_initial)
    top_share = 100.0 * float(second_by_n[top_initial].sum())
    axes[1].text(
        0.98, 0.98,
        "dominant initial sectors: " + ", ".join(str(int(n)) for n in top_sorted)
        + f"\ncombined share: {top_share:.1f}%",
        transform=axes[1].transAxes,
        ha="right", va="top",
        fontsize=FONT_SMALL,
        color="white",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.55, edgecolor="none"),
    )

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.965)
    save_figure(fig, "FigS2b_trajectory_sector_heatmaps")


# =============================================================================
# 7. SUPPLEMENTAL FIGURES
# =============================================================================

def make_sm_figure_time_records(time_df):
    fig, axes = plt.subplots(2, 2, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 5.35))

    ax = axes[0, 0]
    panel_label(ax, "(a)")
    ax.plot(time_df["Jt"], time_df["E"], label=r"$E/J$")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$E/J$")
    ax2 = ax.twinx()
    ax2.plot(time_df["Jt"], time_df["beta_star"], ls="--", label=r"$\beta^\star J$")
    ax2.set_ylabel(r"$\beta^\star J$")

    ax = axes[0, 1]
    panel_label(ax, "(b)")
    ax.plot(time_df["Jt"], time_df["D"], label=r"$D$")
    ax.plot(time_df["Jt"], time_df["Delta_meas"], label=r"$\Delta_{\rm meas}$")
    ax.plot(time_df["Jt"], time_df["Qnc"], label=r"$\mathcal{Q}_{\rm nc}$")
    ax.plot(time_df["Jt"], time_df["mean_sigma"], ls="--", label=r"$\langle\sigma\rangle$")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel("dimensionless")
    ax.legend(ncol=2)

    ax = axes[1, 0]
    panel_label(ax, "(c)")
    for r in range(Q + 1):
        ax.plot(time_df["Jt"], time_df[f"p{r}"], label=fr"$r={r}$")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"undisturbed $p_r$")
    ax.legend(ncol=3)

    ax = axes[1, 1]
    panel_label(ax, "(d)")
    for r in range(Q + 1):
        ax.plot(time_df["Jt"], time_df[f"pseq{r}"] - time_df[f"p{r}"],
                label=fr"$r={r}$")
    ax.axhline(0, lw=0.6)
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel(r"$p_r^{\rm seq}-p_r$")
    ax.legend(ncol=3)

    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.09, top=0.98,
                        wspace=0.37, hspace=0.36)
    save_figure(fig, "FigS1_full_time_resolved")


def make_sm_figure_rare_events(traj_df):
    p = traj_df["prob"].to_numpy()
    sigma = traj_df["sigma"].to_numpy()
    y = traj_df["Y"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.75), gridspec_kw={"wspace": 0.40})

    ax = axes[0]
    panel_label(ax, "(a)")
    bins = np.linspace(np.quantile(sigma, 0.002), np.quantile(sigma, 0.998), 45)
    ax.hist(sigma, bins=bins, weights=p, density=False, histtype="stepfilled", alpha=0.75)
    ax.axvline(0, lw=0.7, ls=":")
    ax.set_xlabel(r"$\sigma_{\rm phys,R}$")
    ax.set_ylabel("trajectory probability")

    ax = axes[1]
    panel_label(ax, "(b)")
    order = np.argsort(sigma)  # rare negative-sigma events first
    cp = np.cumsum(p[order])
    cy = np.cumsum(p[order] * y[order])
    ax.plot(cp, cy)
    ax.plot([0, 1], [0, 1], ls=":", lw=0.7)
    ax.set_xlabel("cumulative trajectory probability")
    ax.set_ylabel(r"cumulative $\langle e^{-\sigma}\rangle$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)

    ax = axes[2]
    panel_label(ax, "(c)")
    second_by_n = traj_df.assign(second=traj_df["prob"] * traj_df["Y"]**2).groupby("n")["second"].sum()
    total = float(second_by_n.sum())
    vals = np.array([second_by_n.get(n, 0.0) / total for n in range(DIM)])
    ax.bar(np.arange(DIM), vals)
    ax.set_xlabel("initial energy index $n$")
    ax.set_ylabel(r"share of $\langle e^{-2\sigma}\rangle$")
    ax.set_xlim(-0.7, DIM - 0.3)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.96)
    save_figure(fig, "FigS2_rare_event_structure")


def make_sm_figure_information_landscape(frontier_raw, frontier_verified):
    raw = frontier_raw.sort_values("D")
    ver = frontier_verified.sort_values("D_achieved")
    N0 = float(ver.iloc[0]["N_1pct_99pct"])

    fig, axes = plt.subplots(1, 3, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.75), gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.semilogx(raw["D"], raw["N"] / N0)
    ax.plot(ver["D_achieved"].replace(0, np.nan), ver["N_1pct_99pct"] / N0, "o")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$N^\star/N_{\rm th}$")

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(raw["D"], 100.0 * raw["saving"])
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("sampling saving (%)")

    ax = axes[2]
    panel_label(ax, "(c)")
    ax.plot(raw["D"], raw["beta"], label=r"$\beta J$")
    for k in range(4):
        ax.plot(raw["D"], raw[f"a{k}"], label=fr"$\alpha_{k}$")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("optimized parameter")
    ax.legend(ncol=2, fontsize=7.4)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.96)
    save_figure(fig, "FigS3_information_sampling_landscape")


def make_sm_figure_robustness(full_grid, endpoint_grid):
    fig, axes = plt.subplots(1, 2, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.75), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.scatter(endpoint_grid["D_th_best"], 100 * endpoint_grid["saving"], s=10, alpha=0.72)
    nom = endpoint_grid[(np.isclose(endpoint_grid["ru"], 1.0)) &
                        (np.isclose(endpoint_grid["rLambda"], 1.0)) &
                        (np.isclose(endpoint_grid["rtau"], 1.0))]
    if len(nom):
        ax.scatter(nom["D_th_best"], 100 * nom["saving"], s=30, marker="*", label="nominal")
        ax.legend()
    ax.set_xlabel(r"endpoint displacement $D_{\rm th}$")
    ax.set_ylabel("best-found saving (%)")

    ax = axes[1]
    panel_label(ax, "(b)")
    groups = full_grid.groupby("s")["saving"]
    svals = np.array(sorted(groups.groups.keys()))
    lo = groups.min().reindex(svals).to_numpy()
    hi = groups.max().reindex(svals).to_numpy()
    med = groups.median().reindex(svals).to_numpy()
    ax.fill_between(svals, 100 * lo, 100 * hi, alpha=0.22, label="5x5x5 envelope")
    ax.plot(svals, 100 * med, lw=0.9, label="grid median")
    nom_t = full_grid[(np.isclose(full_grid["ru"], 1.0)) &
                      (np.isclose(full_grid["rLambda"], 1.0)) &
                      (np.isclose(full_grid["rtau"], 1.0))].sort_values("s")
    ax.plot(nom_t["s"], 100 * nom_t["saving"], ls="--", label="nominal")
    ax.set_xlabel(r"protocol fraction $s=t/\tau$")
    ax.set_ylabel("best-found saving (%)")
    ax.legend()

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.23, top=0.96)
    save_figure(fig, "FigS4_robustness")


def build_compatible_pair(U_endpoint):
    """Deterministic same-(E,R) pair used in the frozen validation."""
    Rin = np.array([
        [np.real(V_I[:, n].conj().T @ P @ V_I[:, n]) for n in range(DIM)]
        for P in P_R
    ])
    C = np.vstack([np.ones(DIM), E_I, Rin])
    b = C @ Q_I
    c2 = E_I**2
    lp_max = linprog(-c2, A_eq=C, b_eq=b, bounds=[(0, None)] * DIM, method="highs")
    lp_min = linprog(c2, A_eq=C, b_eq=b, bounds=[(0, None)] * DIM, method="highs")
    if not (lp_max.success and lp_min.success):
        raise RuntimeError("Compatibility-fiber linear program failed.")
    pA = 0.8 * Q_I + 0.2 * lp_max.x
    pB = 0.8 * Q_I + 0.2 * lp_min.x

    def rho_from_p(p):
        return V_I @ np.diag(p) @ V_I.conj().T

    rhoA_f = U_endpoint @ rho_from_p(pA) @ U_endpoint.conj().T
    rhoB_f = U_endpoint @ rho_from_p(pB) @ U_endpoint.conj().T
    pRA = np.array([np.real(np.trace(P @ rhoA_f)) for P in P_R])
    pRB = np.array([np.real(np.trace(P @ rhoB_f)) for P in P_R])
    gammaA = np.log(pA / Q_I)
    gammaB = np.log(pB / Q_I)
    return pA, pB, pRA, pRB, gammaA, gammaB, C


def make_sm_figure_closure(U_endpoint):
    pA, pB, pRA, pRB, gammaA, gammaB, C = build_compatible_pair(U_endpoint)

    fig, axes = plt.subplots(1, 3, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.75), gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.semilogy(np.arange(DIM), pA, "o-", ms=2.6, label="A")
    ax.semilogy(np.arange(DIM), pB, "s--", ms=2.4, label="B")
    ax.set_xlabel("initial energy index $n$")
    ax.set_ylabel(r"$p_n$")
    ax.legend()

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(np.arange(DIM), gammaA - gammaB)
    ax.axhline(0, lw=0.6)
    ax.set_xlabel("initial energy index $n$")
    ax.set_ylabel(r"$\gamma_n^{(A)}-\gamma_n^{(B)}$")

    ax = axes[2]
    panel_label(ax, "(c)")
    x = np.arange(Q + 1)
    w = 0.36
    ax.bar(x - w/2, pRA, width=w, label="A")
    ax.bar(x + w/2, pRB, width=w, label="B")
    ax.set_xlabel(r"final record $r$")
    ax.set_ylabel("probability")
    ax.set_xticks(x)
    ax.legend()

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.96)
    save_figure(fig, "FigS5_arbitrary_state_closure_counterexample")


def make_sm_figure_closure_precomputed(detail_df, final_record_df):
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45), gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.semilogy(detail_df["n"], detail_df["pA"], "o-", ms=2.6, label="A")
    ax.semilogy(detail_df["n"], detail_df["pB"], "s--", ms=2.4, label="B")
    ax.set_xlabel("initial energy index $n$")
    ax.set_ylabel(r"$p_n$")
    ax.legend()

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(detail_df["n"], detail_df["gammaA"] - detail_df["gammaB"])
    ax.axhline(0, lw=0.6)
    ax.set_xlabel("initial energy index $n$")
    ax.set_ylabel(r"$\gamma_n^{(A)}-\gamma_n^{(B)}$")

    ax = axes[2]
    panel_label(ax, "(c)")
    x = final_record_df["r"].to_numpy()
    w = 0.36
    ax.bar(x - w/2, final_record_df["pR_A_final"], width=w, label="A")
    ax.bar(x + w/2, final_record_df["pR_B_final"], width=w, label="B")
    ax.set_xlabel(r"final record $r$")
    ax.set_ylabel("probability")
    ax.set_xticks(x)
    ax.legend()

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.96)
    save_figure(fig, "FigS5_arbitrary_state_closure_counterexample")


def make_sm_figure_support_deficit(support_df):
    fig, axes = plt.subplots(1, 2, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.75), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.plot(support_df["K_supported"], support_df["IFT_direct_DOP853"], "o-", label="trajectory sum")
    ax.plot(support_df["K_supported"], support_df["IFT_operator_DOP853"], "s--", label="operator form")
    ax.set_xlabel("supported initial sectors $K$")
    ax.set_ylabel(r"$\langle e^{-\sigma_{\rm arb}}\rangle$")
    ax.set_ylim(0.58, 1.01)
    ax.legend()

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.semilogy(support_df["K_supported"], support_df["Lambda_supp_DOP853"], "o-")
    ax.set_xlabel("supported initial sectors $K$")
    ax.set_ylabel(r"support deficit $\Lambda_{\rm supp}$")

    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.23, top=0.96)
    save_figure(fig, "FigS6_support_deficit")


def make_sm_figure_metrology(met_df):
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    panel_label(ax, "(a)")
    labels = ["Gibbs\n$\\beta J=0.12$", "non-Gibbs\nmixture", "canonical-like\nif unknown"]
    criteria = [
        ("N_99pct_all_35_seen_union_bound", "all sectors seen"),
        ("N_for_1pct_99pct_Gaussian_delta", "task-level 1%, 99%"),
        ("N_uniform_abs_gamma_le_0.1_99pct_Chernoff", r"uniform $|\Delta\gamma|\leq0.1$"),
    ]
    x = np.arange(len(labels))
    width = 0.24
    for j, (col, lab) in enumerate(criteria):
        ax.bar(x + (j - 1) * width, met_df[col].to_numpy(), width=width, label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("calibration shots")
    ax.set_xticks(x, labels)
    ax.legend(ncol=1, loc="upper left")
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.25, top=0.96)
    save_figure(fig, "FigS7_initial_calibration_metrology")


# =============================================================================
# 8. TABLES
# =============================================================================

def fmt_vector(v, nd=6):
    return "(" + ", ".join(f"{x:.{nd}g}" for x in v) + ")"


def write_table_s1(endpoint):
    rows = [
        ("Hilbert-space dimension", f"{DIM}"),
        (r"$L,Q$", f"{L}, {Q}"),
        (r"$U_Q/J$", f"{UQ:.9f}"),
        (r"$\Lambda_0/J$", f"{LAMBDA0:.1f}"),
        (r"$\beta_iJ$", f"{BETA_I:.6f}"),
        (r"$J\tau$", f"{TAU:.1f}"),
        (r"$E_B/J$", f"{E_B:.12f}"),
        (r"$S_B/k_B$", f"{S_I:.12f}"),
        (r"$p_r(B)$", fmt_vector(P_B, 7)),
        (r"$E_{C^-}/J$", f"{endpoint['E']:.12f}"),
        (r"$W_{\rm on}/J$", f"{endpoint['E'] - E_B:.12f}"),
        (r"$p_r(C^-)$", fmt_vector(endpoint["p"], 7)),
        (r"$p_r^{\rm seq}(C^-)$", fmt_vector(endpoint["pseq"], 7)),
        (r"$\beta_f^\star J$", f"{endpoint['beta']:.12f}"),
        (r"$\alpha_r$ (gauge $\alpha_4=0$)", fmt_vector(endpoint["alpha"], 7)),
        (r"$D(\rho_f\Vert\bar\rho_f)$", f"{endpoint['D']:.12f}"),
        (r"$\Delta_{\rm meas}$", f"{endpoint['Delta_meas']:.12f}"),
        (r"$\mathcal{Q}_{\rm nc}$", f"{endpoint['Qnc']:.12f}"),
        (r"$\langle\sigma_{\rm phys,R}\rangle$", f"{endpoint['mean_sigma']:.12f}"),
    ]
    pd.DataFrame(rows, columns=["Quantity", "Value"]).to_csv(OUT_DIR / "TableS1_model_endpoint.csv", index=False)

    tex = [
        r"\begin{table}[t]",
        r"\caption{Nominal Bose--Hubbard model and endpoint quantities.}",
        r"\label{tab:model-endpoint}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lc}",
        r"Quantity & Value \\",
    ]
    row_end = r"\\"
    for q, v in rows:
        tex.append(f"{q} & {v} {row_end}")
    tex += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"]
    (OUT_DIR / "TableS1_model_endpoint.tex").write_text("\n".join(tex), encoding="utf-8")


def write_table_s2(arb_fullrank, time_df):
    can = arb_fullrank.loc[arb_fullrank["state"] == "canonical"].iloc[0]
    max_time_resid = float(time_df["maxent_resid"].max())
    rows = [
        (r"$E_f/J$", can["Ef_DOP853"], can["Ef_Magnus_dt0.01"], can["abs_Ef_solver_diff"]),
        (r"$D$", can["D_DOP853"], can["D_Magnus_dt0.01"], can["abs_D_solver_diff"]),
        (r"$\Delta_{\rm meas}$", can["Delta_meas_DOP853"], can["Delta_meas_Magnus_dt0.01"], abs(can["Delta_meas_DOP853"] - can["Delta_meas_Magnus_dt0.01"])),
        (r"$\mathcal{Q}_{\rm nc}$", can["Qnc_DOP853"], can["Qnc_Magnus_dt0.01"], can["abs_Qnc_solver_diff"]),
        (r"$\langle\sigma\rangle$", can["mean_sigma_DOP853"], can["mean_sigma_Magnus_dt0.01"], abs(can["mean_sigma_DOP853"] - can["mean_sigma_Magnus_dt0.01"])),
        (r"$\langle e^{-\sigma}\rangle$", can["IFT_DOP853"], can["IFT_Magnus_dt0.01"], abs(can["IFT_DOP853"] - can["IFT_Magnus_dt0.01"])),
    ]
    out = pd.DataFrame(rows, columns=["Quantity", "DOP853", "Magnus_dt0.01", "Abs_difference"])
    out.to_csv(OUT_DIR / "TableS2_numerical_validation.csv", index=False)

    tex = [
        r"\begin{table}[t]",
        r"\caption{Independent numerical validation of the nominal endpoint. The last line below the table gives the largest MaxEnt constraint residual on the time-resolved scan.}",
        r"\label{tab:numerical-validation}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lccc}",
        r"Quantity & DOP853 & Magnus & $|\Delta|$ \\",
    ]
    row_end = r"\\"
    for q, a, b, dff in rows:
        tex.append(f"{q} & {a:.12g} & {b:.12g} & {dff:.3g} {row_end}")
    tex += [
        r"\end{tabular}",
        r"\end{ruledtabular}",
        rf"\par\smallskip Maximum time-scan MaxEnt residual: ${max_time_resid:.3g}$.",
        r"\end{table}",
    ]
    (OUT_DIR / "TableS2_numerical_validation.tex").write_text("\n".join(tex), encoding="utf-8")


def write_table_s3(arb_fullrank, closure_df, support_df):
    cols = ["state", "Ei_DOP853", "Ef_DOP853", "D_DOP853", "Delta_meas_DOP853",
            "Qnc_DOP853", "mean_sigma_DOP853", "IFT_DOP853"]
    arb_fullrank[cols].to_csv(OUT_DIR / "TableS3_arbitrary_initial_states.csv", index=False)
    support_df.to_csv(OUT_DIR / "TableS3b_support_deficit.csv", index=False)

    tex = [
        r"\begin{table*}[t]",
        r"\caption{Full-rank arbitrary-initial-state validation. All entries use the DOP853 propagation; the independent Magnus values are reported in Table~\ref{tab:numerical-validation} and the data archive.}",
        r"\label{tab:arbitrary-states}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lrrrrrrr}",
        r"Preparation & $E_i/J$ & $E_f/J$ & $D$ & $\Delta_{\rm meas}$ & $\mathcal{Q}_{\rm nc}$ & $\langle\sigma\rangle$ & IFT \\",
    ]
    row_end = r"\\"
    for _, row in arb_fullrank.iterrows():
        state_name = row["state"].replace("_", r"\_")
        tex.append(
            f"{state_name} & {row['Ei_DOP853']:.6g} & {row['Ef_DOP853']:.6g} & "
            f"{row['D_DOP853']:.6g} & {row['Delta_meas_DOP853']:.6g} & {row['Qnc_DOP853']:.6g} & "
            f"{row['mean_sigma_DOP853']:.6g} & {row['IFT_DOP853']:.12g} {row_end}")
    tex += [r"\end{tabular}", r"\end{ruledtabular}"]

    # Add the most compact closure/coherence facts as prose below the tabular.
    lookup = {row.metric: row.value for row in closure_df.itertuples()}
    if "initial_energy_distribution_TV" in lookup:
        tex.append(
            rf"\par\smallskip The explicit same-$(E,R)$ pair has initial energy-distribution TV distance "
            rf"${lookup['initial_energy_distribution_TV']:.6g}$ and $\|\Gamma_A-\Gamma_B\|_\infty="
            rf"{lookup.get('Gamma_operator_sup_difference', float('nan')):.6g}$.")
    if "trace_distance_unmeasured_vs_TPM_endpoint" in lookup:
        tex.append(
            rf" The coherent canonical-population test gives trace distance "
            rf"${lookup['trace_distance_unmeasured_vs_TPM_endpoint']:.6g}$ between the unmeasured and TPM states.")
    tex.append(r"\end{table*}")
    (OUT_DIR / "TableS3_arbitrary_initial_states.tex").write_text("\n".join(tex), encoding="utf-8")



# =============================================================================
# 9. FREE-ENERGY ESTIMATION FROM THE EXACT REFERENCE FAMILY
# =============================================================================

# IMPORTANT CHANGE OF OBJECTIVE
# -----------------------------
# The original script compared the sampling cost of different exact family
# members as different normalized fluctuation relations.  The revised analysis
# fixes ONE physical target throughout: the ordinary Jarzynski free-energy ratio
#
#       R_J = exp(-beta_i Delta F) = Z_f(beta_i) / Z_i.
#
# The endpoint-matching thermodynamic member, and nearby exact family members,
# are now used as zero-mean control variates for the SAME estimator of R_J.
# Consequently, every sampling-cost bar and every point on the new frontier is
# an apples-to-apples estimate of the same Delta F.
#
# Operational caveat: a control member is usable only if its normalization
# Z_f^seq(theta)/Z_i is available independently.  The thermodynamic member has
# this normalization from the independently calibrated coarse endpoint model.
# The local displaced-reference frontier below assumes the same for nearby
# admissible references.  We intentionally keep the frontier local around the
# thermodynamic endpoint and never use the Jarzynski member itself as a control,
# because its normalization is precisely the unknown target R_J.

# Write revised outputs to a separate folder so the previous figures are never
# silently overwritten.
OUT_DIR = HERE / "PRL_figures_tables_free_energy_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# High-contrast, color-blind-friendly palette (Okabe-Ito family).
C_BLACK = "#000000"
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_SKY = "#56B4E9"
C_MAGENTA = "#CC79A7"
C_GRAY = "#666666"
C_YELLOW = "#E69F00"

# Main controlled-fidelity budget used for the four-bar comparison in Fig. 2(a).
CONTROL_DELTA_BUDGET = 0.01

# The family frontier is intentionally local.  This keeps the auxiliary
# references tied to the measured thermodynamic endpoint and avoids the
# circular Jarzynski-as-its-own-control limit.
FRONTIER_DELTAS = np.array([
    0.0, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 3e-2, 4e-2, 5e-2,
], dtype=float)

# Numerical finite-difference scale for the local information geometry.
LOCAL_GEOM_STEP = 3e-4


def final_canonical_free_energy():
    """Exact model value used only to benchmark estimator accuracy/cost.

    In an experiment Delta F is the unknown target.  The code knows it because
    the finite Bose-Hubbard benchmark is exactly diagonalizable; this lets us
    evaluate bias, variance and finite-confidence rates of proposed estimators.
    """
    _, _, _, _, logZ_f_J = canonical_state(H_F, BETA_I)
    log_RJ = logZ_f_J - LOGZ_I
    RJ = math.exp(log_RJ)
    deltaF = -log_RJ / BETA_I
    return RJ, deltaF, logZ_f_J


def thermodynamic_theta(endpoint):
    """Gauge-fixed theta=(beta,alpha0,...,alpha3); alpha4=0."""
    return np.r_[float(endpoint["beta"]), np.asarray(endpoint["alpha"][:4], dtype=float)]


def exact_reference_displacement(theta, theta_th, moments_th, logZ_th):
    """D(rho_th || rho_theta) for the joint-exponential MaxEnt family.

    With Phi(theta)=ln Z(theta),
      D = Phi(theta)-Phi(theta_th)+(theta-theta_th).moments_th.
    """
    theta = np.asarray(theta, dtype=float)
    _, _, logZ, _ = maxent_constraints(theta, H_F)
    D = float(logZ - logZ_th + np.dot(theta - theta_th, moments_th))
    # Small negative values can occur at machine precision near theta_th.
    return max(0.0, D)


def _tilted_cgf_and_mean(t, y, logp):
    """Return K(t)=ln E[e^{tY}] and K'(t), stably."""
    z = logp + t * y
    zmax = float(np.max(z))
    w = np.exp(z - zmax)
    sw = float(np.sum(w))
    K = zmax + math.log(sw)
    Kp = float(np.dot(w, y) / sw)
    return K, Kp


def cramer_rate_fast(y, p, a, upper):
    """Exact finite-support Cramer rate via the tilted-mean root.

    This is numerically faster and more accurate than repeatedly maximizing the
    Legendre transform with a nested generic scalar optimizer.
    """
    from scipy.optimize import brentq

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    logp = np.log(p)
    mean = float(np.dot(p, y))

    if upper:
        if a <= mean:
            return 0.0
        if a >= float(np.max(y)):
            return np.inf
        lo, hi = 0.0, 1e-10
        while _tilted_cgf_and_mean(hi, y, logp)[1] < a:
            hi *= 2.0
            if hi > 1e8:
                raise RuntimeError("Could not bracket upper-tail Cramer saddle.")
    else:
        if a >= mean:
            return 0.0
        if a <= float(np.min(y)):
            return np.inf
        lo, hi = -1e-10, 0.0
        while _tilted_cgf_and_mean(lo, y, logp)[1] > a:
            lo *= 2.0
            if lo < -1e8:
                raise RuntimeError("Could not bracket lower-tail Cramer saddle.")

    root = brentq(
        lambda t: _tilted_cgf_and_mean(t, y, logp)[1] - a,
        lo, hi, xtol=1e-14, rtol=1e-12, maxiter=150,
    )
    K, _ = _tilted_cgf_and_mean(root, y, logp)
    return max(0.0, float(root * a - K))


def chernoff_sample_size_fast(y, p, eps=REL_TOL, eta=ETA):
    """Two-sided Cramer-Chernoff benchmark for a unit-mean variable."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    mean = float(np.dot(p, y))
    if abs(mean - 1.0) > 2e-8:
        raise ValueError(f"Controlled estimator is not unit mean: {mean:.12g}")

    rp = cramer_rate_fast(y, p, 1.0 + eps, upper=True)
    rm = cramer_rate_fast(y, p, 1.0 - eps, upper=False)
    rate = min(rp, rm)
    if rate <= 0.0:
        return np.inf, rp, rm
    return math.log(2.0 / eta) / rate, rp, rm


def jarzynski_and_control_weights(theta, traj_df, Rmr, RJ):
    """Return normalized Jarzynski and exact-family control weights.

    YJ has mean 1 but is used only internally for model-conditioned accuracy
    benchmarking.  The actual raw estimator uses XJ=exp(-beta_i W), which does
    not require knowing Delta F.

    Ytheta=exp(-sigma_theta) has exactly known mean 1 when the reference
    normalization is independently calibrated.
    """
    p = traj_df["prob"].to_numpy(dtype=float)
    p = p / p.sum()
    W = traj_df["W"].to_numpy(dtype=float)
    XJ = np.exp(-BETA_I * W)
    YJ = XJ / RJ
    Ytheta, sigma_theta = family_Y_for_theta(theta, traj_df, Rmr)
    return p, XJ, YJ, Ytheta, sigma_theta


def variance_optimal_control_lambda(YJ, Ytheta, p):
    """Dimensionless lambda=(raw control coefficient)/R_J."""
    dJ = YJ - 1.0
    dT = Ytheta - 1.0
    varT = float(np.dot(p, dT * dT))
    if varT < 1e-30:
        return 0.0
    return float(np.dot(p, dJ * dT) / varT)


def controlled_unit_weight(YJ, Ytheta, lam):
    """Unit-mean controlled variable for relative-error benchmarking.

    In raw experimental units the unbiased estimator is
      XJ - c (Ytheta-1),   c = lam * R_J.
    The code divides by the exact benchmark R_J only to express the confidence
    event as a relative error around one.
    """
    return YJ - float(lam) * (Ytheta - 1.0)


def finite_confidence_optimal_lambda(theta, traj_df, Rmr, RJ, lambda_hint=None):
    """Best scalar control coefficient for the two-sided finite-confidence cost."""
    p, _, YJ, Ytheta, _ = jarzynski_and_control_weights(theta, traj_df, Rmr, RJ)
    lam_var = variance_optimal_control_lambda(YJ, Ytheta, p)
    if lambda_hint is None:
        lambda_hint = lam_var

    # Coarse deterministic scan prevents the bounded scalar optimizer from
    # locking onto a poor local kink when the active Cramer tail switches.
    L = max(8.0, 4.0 * abs(lam_var) + 4.0, 2.0 * abs(lambda_hint) + 4.0)
    grid = np.linspace(-L, L, 41)

    def logN(lam):
        G = controlled_unit_weight(YJ, Ytheta, lam)
        N, _, _ = chernoff_sample_size_fast(G, p)
        return math.log(N)

    vals = np.array([logN(lam) for lam in grid])
    k = int(np.argmin(vals))
    lo = float(grid[max(0, k - 1)])
    hi = float(grid[min(len(grid) - 1, k + 1)])
    if hi <= lo:
        return float(grid[k]), float(math.exp(vals[k]))

    res = minimize_scalar(
        logN, bounds=(lo, hi), method="bounded",
        options={"xatol": 1e-8},
    )
    return float(res.x), float(math.exp(res.fun))


def control_metrics(theta, lam, traj_df, Rmr, RJ):
    """All sampling metrics for one admissible exact control reference."""
    p, XJ, YJ, Ytheta, sigma = jarzynski_and_control_weights(theta, traj_df, Rmr, RJ)
    G = controlled_unit_weight(YJ, Ytheta, lam)
    N, rp, rm = chernoff_sample_size_fast(G, p)
    varG = float(np.dot(p, (G - 1.0) ** 2))
    varJ = float(np.dot(p, (YJ - 1.0) ** 2))
    varTheta = float(np.dot(p, (Ytheta - 1.0) ** 2))
    cov = float(np.dot(p, (YJ - 1.0) * (Ytheta - 1.0)))
    corr = cov / math.sqrt(varJ * varTheta) if varJ > 0 and varTheta > 0 else np.nan
    return {
        "N": N,
        "I_plus": rp,
        "I_minus": rm,
        "variance": varG,
        "variance_jarzynski": varJ,
        "variance_control": varTheta,
        "correlation": corr,
        "lambda": float(lam),
        "raw_c": float(lam * RJ),
        "G": G,
        "YJ": YJ,
        "Ytheta": Ytheta,
        "XJ": XJ,
        "sigma_theta": sigma,
        "p": p,
    }


def _profiled_logN(theta, traj_df, Rmr, RJ, lambda_hint=None):
    lam, N = finite_confidence_optimal_lambda(theta, traj_df, Rmr, RJ, lambda_hint=lambda_hint)
    return math.log(N), lam


def local_information_geometry(theta_th, traj_df, Rmr, RJ):
    """BKM metric and local square-root coefficient for the Delta-F frontier."""
    h = LOCAL_GEOM_STEP
    moments_th, _, logZ_th, _ = maxent_constraints(theta_th, H_F)
    lam0, N0 = finite_confidence_optimal_lambda(theta_th, traj_df, Rmr, RJ)

    grad = np.zeros_like(theta_th, dtype=float)
    for i in range(len(theta_th)):
        tp = theta_th.copy(); tp[i] += h
        tm = theta_th.copy(); tm[i] -= h
        fp, _ = _profiled_logN(tp, traj_df, Rmr, RJ, lambda_hint=lam0)
        fm, _ = _profiled_logN(tm, traj_df, Rmr, RJ, lambda_hint=lam0)
        grad[i] = (fp - fm) / (2.0 * h)

    # Phi=ln Z, grad Phi=-moments, hence Hess Phi=-d moments/d theta.
    G = np.zeros((len(theta_th), len(theta_th)), dtype=float)
    for j in range(len(theta_th)):
        tp = theta_th.copy(); tp[j] += h
        tm = theta_th.copy(); tm[j] -= h
        mp, _, _, _ = maxent_constraints(tp, H_F)
        mm, _, _, _ = maxent_constraints(tm, H_F)
        G[:, j] = -(mp - mm) / (2.0 * h)
    G = 0.5 * (G + G.T)

    eig = np.linalg.eigvalsh(G)
    if np.min(eig) <= 0:
        raise RuntimeError(f"Local BKM metric is not positive definite: {eig}")

    Ginv_grad = np.linalg.solve(G, grad)
    q = float(np.dot(grad, Ginv_grad))
    C_sqrt = math.sqrt(2.0 * q)
    natural_direction = -Ginv_grad
    return {
        "G": G,
        "grad_logN": grad,
        "C_sqrt": C_sqrt,
        "natural_direction": natural_direction,
        "lambda0": lam0,
        "N0": N0,
        "moments_th": moments_th,
        "logZ_th": logZ_th,
    }


def _boundary_point_along_direction(theta_th, direction, delta, moments_th, logZ_th):
    """Find theta_th+s*direction with exact D=delta."""
    from scipy.optimize import brentq
    if delta <= 0:
        return theta_th.copy()
    direction = np.asarray(direction, dtype=float)

    def f(s):
        return exact_reference_displacement(
            theta_th + s * direction, theta_th, moments_th, logZ_th
        ) - delta

    hi = 1.0
    while f(hi) < 0.0 and hi < 128.0:
        hi *= 2.0
    if f(hi) < 0.0:
        raise RuntimeError("Could not reach requested information budget along start direction.")
    s = brentq(f, 0.0, hi, xtol=1e-12, rtol=1e-10)
    return theta_th + s * direction


def optimize_finite_confidence_reference(
    delta, theta_th, geom, traj_df, Rmr, RJ, previous=None
):
    """Best-found Delta-F control reference with D(rho_th||rho_theta)<=delta.

    The scalar control coefficient and the five gauge-fixed reference parameters
    are optimized together.  Two deterministic starts are used: continuation
    from the previous budget and the local natural-gradient direction.
    """
    moments_th = geom["moments_th"]
    logZ_th = geom["logZ_th"]
    direction = geom["natural_direction"]

    if delta <= 0:
        lam, N = finite_confidence_optimal_lambda(theta_th, traj_df, Rmr, RJ)
        return theta_th.copy(), lam, N

    theta_nat = _boundary_point_along_direction(
        theta_th, direction, delta, moments_th, logZ_th
    )
    lam_nat, _ = finite_confidence_optimal_lambda(
        theta_nat, traj_df, Rmr, RJ, lambda_hint=geom["lambda0"]
    )

    starts = [np.r_[theta_nat, lam_nat]]
    if previous is not None:
        starts.insert(0, np.asarray(previous, dtype=float))

    p = traj_df["prob"].to_numpy(dtype=float)
    p = p / p.sum()
    W = traj_df["W"].to_numpy(dtype=float)
    YJ = np.exp(-BETA_I * W) / RJ

    def objective(z):
        theta = np.asarray(z[:5], dtype=float)
        lam = float(z[5])
        Ytheta, _ = family_Y_for_theta(theta, traj_df, Rmr)
        G = controlled_unit_weight(YJ, Ytheta, lam)
        N, _, _ = chernoff_sample_size_fast(G, p)
        return math.log(N)

    def constraint(z):
        return delta - exact_reference_displacement(
            z[:5], theta_th, moments_th, logZ_th
        )

    bounds = [(-1.0, 5.0)] + [(-5.0, 5.0)] * 4 + [(-30.0, 30.0)]
    # A generic constrained optimizer is allowed to move only if it actually
    # improves a feasible deterministic start.  This guarantees that the
    # best-found cost cannot get worse merely because SLSQP took a poor step.
    best_x = None
    best_fun = np.inf
    for start in starts:
        Dstart = exact_reference_displacement(
            start[:5], theta_th, moments_th, logZ_th
        )
        if Dstart <= delta + 2e-6:
            fstart = objective(start)
            if fstart < best_fun:
                best_fun = fstart
                best_x = np.asarray(start, dtype=float).copy()

        res = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraint}],
            options={"maxiter": 180, "ftol": 2e-10, "disp": False},
        )
        Dres = exact_reference_displacement(
            res.x[:5], theta_th, moments_th, logZ_th
        )
        if res.success and Dres <= delta + 2e-6 and res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = np.asarray(res.x, dtype=float).copy()

    if best_x is None:
        warnings.warn(
            f"Finite-confidence reference optimization failed at delta={delta:g}; "
            "using the natural-gradient boundary start."
        )
        theta = theta_nat
        lam, N = finite_confidence_optimal_lambda(theta, traj_df, Rmr, RJ)
        return theta, lam, N

    return np.asarray(best_x[:5], dtype=float), float(best_x[5]), float(math.exp(best_fun))


def optimize_variance_reference(delta, theta_th, geom, traj_df, Rmr, RJ, start_theta=None):
    """Variance-optimal admissible control reference at the same delta budget."""
    moments_th = geom["moments_th"]
    logZ_th = geom["logZ_th"]
    p = traj_df["prob"].to_numpy(dtype=float)
    p = p / p.sum()
    W = traj_df["W"].to_numpy(dtype=float)
    YJ = np.exp(-BETA_I * W) / RJ

    def profiled_variance(theta):
        Ytheta, _ = family_Y_for_theta(theta, traj_df, Rmr)
        lam = variance_optimal_control_lambda(YJ, Ytheta, p)
        G = controlled_unit_weight(YJ, Ytheta, lam)
        return float(np.dot(p, (G - 1.0) ** 2)), lam

    if start_theta is None:
        start_theta = _boundary_point_along_direction(
            theta_th, geom["natural_direction"], delta, moments_th, logZ_th
        )

    def objective(theta):
        var, _ = profiled_variance(theta)
        return math.log(max(var, 1e-300))

    def constraint(theta):
        return delta - exact_reference_displacement(
            theta, theta_th, moments_th, logZ_th
        )

    bounds = [(-1.0, 5.0)] + [(-5.0, 5.0)] * 4
    res = minimize(
        objective,
        start_theta,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraint}],
        options={"maxiter": 350, "ftol": 1e-12, "disp": False},
    )
    theta = np.asarray(res.x, dtype=float)
    var, lam = profiled_variance(theta)
    return theta, float(lam), float(var), bool(res.success)


def build_free_energy_frontier(theta_th, traj_df, Rmr, RJ):
    """Compute the revised endpoint-information / Delta-F sampling frontier."""
    geom = local_information_geometry(theta_th, traj_df, Rmr, RJ)

    # Guard against the circular limiting point.  The local frontier must not
    # approach theta_J=(beta_i,0), whose normalization is the unknown R_J.
    theta_J = np.array([BETA_I, 0.0, 0.0, 0.0, 0.0])
    D_J = exact_reference_displacement(
        theta_J, theta_th, geom["moments_th"], geom["logZ_th"]
    )
    if float(np.max(FRONTIER_DELTAS)) >= 0.25 * D_J:
        warnings.warn(
            "FRONTIER_DELTAS extends too far toward the Jarzynski member. "
            "Keep the control-reference optimization local to the measured endpoint."
        )

    rows = []
    previous = None
    for delta in FRONTIER_DELTAS:
        theta, lam, N = optimize_finite_confidence_reference(
            float(delta), theta_th, geom, traj_df, Rmr, RJ, previous=previous
        )
        D = exact_reference_displacement(
            theta, theta_th, geom["moments_th"], geom["logZ_th"]
        )
        metrics = control_metrics(theta, lam, traj_df, Rmr, RJ)
        row = {
            "delta_budget": float(delta),
            "D_achieved": D,
            "N_DeltaF": N,
            "N_over_Nth": N / geom["N0"],
            "saving_vs_thermo": 1.0 - N / geom["N0"],
            "lambda": lam,
            "raw_c": lam * RJ,
            "variance": metrics["variance"],
            "correlation": metrics["correlation"],
            "beta": theta[0],
            "a0": theta[1],
            "a1": theta[2],
            "a2": theta[3],
            "a3": theta[4],
        }
        rows.append(row)
        previous = np.r_[theta, lam]

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "free_energy_information_sampling_frontier.csv", index=False)
    return df, geom, D_J


def interpolate_frontier_solution(frontier, delta):
    """Return the computed row nearest a requested budget."""
    idx = int(np.argmin(np.abs(frontier["delta_budget"].to_numpy() - delta)))
    return frontier.iloc[idx]


# =============================================================================
# 10. REVISED MAIN FIGURES
# =============================================================================


def make_main_figure_1_updated(time_df, endpoint):
    """Original physical theorem figure, restyled with explicit high contrast."""
    fig = plt.figure(figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.85, 1.35], wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    panel_label(ax, "(a)")
    boxes = [
        (0.02, 0.44, 0.27, 0.22, "Gibbs $B$\n" + r"$\rho_B$"),
        (0.365, 0.44, 0.27, 0.22, "isolated drive\n" + r"$H_i\to H_f$"),
        (0.71, 0.44, 0.27, 0.22, "endpoint $C^{-}$\n" + r"$E_f,p_r$"),
    ]
    for x, y, w, h, txt in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, transform=ax.transAxes,
            boxstyle="round,pad=0.018", facecolor="0.96",
            edgecolor=C_BLACK, linewidth=0.8, clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT_SMALL, linespacing=1.22)
    ax.annotate("", xy=(0.36, 0.55), xytext=(0.295, 0.55), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C_BLACK))
    ax.annotate("", xy=(0.705, 0.55), xytext=(0.64, 0.55), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C_BLACK))
    ax.text(0.50, 0.20,
            r"TPM: $E_n^i\to E_m^f\to r$" + "\n" + r"$W_{mn}=E_m^f-E_n^i$",
            transform=ax.transAxes, ha="center", va="center", fontsize=FONT_SMALL + 0.3)
    ax.text(0.50, 0.84, r"retained macrostate: $\bar\rho_{E,R}$",
            transform=ax.transAxes, ha="center", va="center", fontsize=FONT_SMALL + 0.6)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b)")
    rvals = np.arange(Q + 1)
    width = 0.37
    ax.bar(rvals - width / 2, endpoint["p"], width=width, color=C_BLUE, label=r"$p_r$")
    ax.bar(rvals + width / 2, endpoint["pseq"], width=width, color=C_ORANGE,
           label=r"$p_r^{\rm seq}$")
    ax.set_xlabel(r"record $r=n_R$")
    ax.set_ylabel("probability")
    ax.set_xticks(rvals)
    ax.set_ylim(0, 0.34)
    ax.legend(loc="upper right", ncol=1)

    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "(c)")
    ax.plot(time_df["Jt"], time_df["D"], color=C_BLUE,
            label=r"$D(\rho_t\Vert\bar\rho_t)$")
    ax.plot(time_df["Jt"], time_df["mean_sigma"], color=C_ORANGE, ls="--",
            label=r"$\langle\sigma_{\rm phys,R}\rangle$")
    ax.set_xlabel(r"$Jt$")
    ax.set_ylabel("dimensionless")
    ax.set_xlim(0, TAU)
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))

    ins = inset_axes(ax, width="47%", height="48%", loc="center", borderpad=0.8)
    ins.plot(time_df["Jt"], time_df["Delta_meas"], color=C_GREEN,
             label=r"$\Delta_{\rm meas}$")
    ins.plot(time_df["Jt"], time_df["Qnc"], color=C_MAGENTA,
             label=r"$\mathcal{Q}_{\rm nc}$")
    ins.set_xlim(0, TAU)
    ins.tick_params(labelsize=7.2)
    ins.legend(fontsize=7.0, loc="upper left")

    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.19, top=0.96)
    save_figure(fig, "Fig1_PRL_physical_work_record")


def sector_variance_contribution(traj_df, G, normalization):
    """Per-initial-sector variance contribution on a common normalization."""
    p = traj_df["prob"].to_numpy(dtype=float)
    n = traj_df["n"].to_numpy(dtype=int)
    contrib = p * (G - 1.0) ** 2
    out = np.bincount(n, weights=contrib, minlength=DIM).astype(float)
    return out / float(normalization)


def make_main_figure_2_free_energy(frontier, theta_th, geom, traj_df, Rmr, RJ, deltaF):
    """Main new result: all sampling costs estimate the SAME Delta F."""
    p = traj_df["prob"].to_numpy(dtype=float)
    p = p / p.sum()
    W = traj_df["W"].to_numpy(dtype=float)
    YJ = np.exp(-BETA_I * W) / RJ
    N_J, _, _ = chernoff_sample_size_fast(YJ, p)

    # Exact thermodynamic control reference.
    lam_th, N_th = finite_confidence_optimal_lambda(theta_th, traj_df, Rmr, RJ)
    met_th = control_metrics(theta_th, lam_th, traj_df, Rmr, RJ)

    # Same fidelity budget for the variance- and finite-confidence-optimized bars.
    row_fc = interpolate_frontier_solution(frontier, CONTROL_DELTA_BUDGET)
    theta_fc = np.array([row_fc["beta"], row_fc["a0"], row_fc["a1"], row_fc["a2"], row_fc["a3"]])
    lam_fc = float(row_fc["lambda"])
    N_fc = float(row_fc["N_DeltaF"])

    theta_var, lam_var, _, ok_var = optimize_variance_reference(
        CONTROL_DELTA_BUDGET, theta_th, geom, traj_df, Rmr, RJ,
        start_theta=theta_fc,
    )
    if not ok_var:
        warnings.warn("Variance-reference optimization did not report success; using best returned point.")
    met_var = control_metrics(theta_var, lam_var, traj_df, Rmr, RJ)
    N_var = met_var["N"]

    # Save the exact numerical comparison used by the figure.
    comparison = pd.DataFrame([
        {
            "reference": "direct_Jarzynski",
            "delta": np.nan,
            "lambda": 0.0,
            "variance": float(np.dot(p, (YJ - 1.0) ** 2)),
            "N": N_J,
            "saving_vs_Jarzynski": 0.0,
        },
        {
            "reference": "thermodynamic_control",
            "delta": 0.0,
            "lambda": lam_th,
            "variance": met_th["variance"],
            "N": N_th,
            "saving_vs_Jarzynski": 1.0 - N_th / N_J,
        },
        {
            "reference": "variance_optimized_control",
            "delta": exact_reference_displacement(theta_var, theta_th, geom["moments_th"], geom["logZ_th"]),
            "lambda": lam_var,
            "variance": met_var["variance"],
            "N": N_var,
            "saving_vs_Jarzynski": 1.0 - N_var / N_J,
        },
        {
            "reference": "finite_confidence_optimized_control",
            "delta": float(row_fc["D_achieved"]),
            "lambda": lam_fc,
            "variance": control_metrics(theta_fc, lam_fc, traj_df, Rmr, RJ)["variance"],
            "N": N_fc,
            "saving_vs_Jarzynski": 1.0 - N_fc / N_J,
        },
    ])
    comparison.to_csv(OUT_DIR / "main_DeltaF_sampling_costs.csv", index=False)

    fig = plt.figure(figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.18))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.96, 1.10, 1.08], wspace=0.44)

    # (a) Same-target sampling costs.
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "(a)")
    labels = ["Jarz.", "thermo.\nCV", "var.\nCV", "conf.\nCV"]
    vals = np.array([N_J, N_th, N_var, N_fc]) / 1e9
    colors = [C_GRAY, C_BLUE, C_GREEN, C_ORANGE]
    bars = ax.bar(np.arange(4), vals, width=0.68, color=colors)
    ax.set_ylabel(r"TPM runs $N$ ($10^9$)")
    ax.set_xticks(np.arange(4), labels)
    ax.tick_params(axis="x", labelsize=FONT_SMALL)
    ax.set_ylim(0, max(vals) * 1.20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.035 * max(vals),
                f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SMALL)
    ax.text(0.98, 0.98, r"same $\Delta F$", transform=ax.transAxes,
            ha="right", va="top", fontsize=FONT_SMALL + 0.1)

    # (b) Information-sampling frontier for Delta F.
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b)")
    ax.plot(frontier["D_achieved"], frontier["N_over_Nth"], color=C_BLUE,
            marker="o", ms=3.6, lw=1.35, label=r"best-found $\Delta F$ frontier")
    dsmall = np.linspace(0.0, min(0.012, float(frontier["D_achieved"].max())), 200)
    ax.plot(dsmall, 1.0 - geom["C_sqrt"] * np.sqrt(dsmall), color=C_ORANGE,
            ls="--", lw=1.2,
            label=rf"$1-{geom['C_sqrt']:.2f}\sqrt{{\delta}}$")
    ax.axvline(CONTROL_DELTA_BUDGET, color=C_BLACK, lw=0.8, ls=":")
    ax.set_xlabel(r"reference displacement $\delta$")
    ax.set_ylabel(r"$N_{\Delta F}^\star(\delta)/N_{\rm th}$")
    ax.set_xlim(0.0, float(frontier["delta_budget"].max()) * 1.02)
    ymin = max(0.45, float(frontier["N_over_Nth"].min()) - 0.04)
    ax.set_ylim(ymin, 1.02)
    ax.legend(loc="upper right", fontsize=FONT_SMALL - 0.2)

    # (c) Rare-event suppression in the actual Delta-F estimator.
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "(c)")
    varJ_total = float(np.dot(p, (YJ - 1.0) ** 2))
    share_J = sector_variance_contribution(traj_df, YJ, varJ_total)
    share_th = sector_variance_contribution(traj_df, met_th["G"], varJ_total)
    nvals = np.arange(DIM)
    floor = 1e-10
    ax.semilogy(nvals, np.maximum(share_J, floor), color=C_GRAY, marker="o", ms=2.8,
                lw=1.0, label="Jarzynski")
    ax.semilogy(nvals, np.maximum(share_th, floor), color=C_BLUE, marker="s", ms=2.7,
                lw=1.0, label="thermo. control")
    ax.set_xlabel(r"initial energy sector $n$")
    ax.set_ylabel(r"variance contribution $/\,\mathrm{Var}_{J}$")
    ax.set_xlim(-0.5, DIM - 0.5)
    ax.legend(loc="upper left", fontsize=FONT_SMALL - 0.1)

    fig.subplots_adjust(left=0.064, right=0.995, bottom=0.23, top=0.965)
    save_figure(fig, "Fig2_PRL_DeltaF_control_variates")

    return comparison, {
        "N_J": N_J,
        "N_th": N_th,
        "N_var": N_var,
        "N_fc": N_fc,
        "lambda_th": lam_th,
        "theta_var": theta_var,
        "lambda_var": lam_var,
        "theta_fc": theta_fc,
        "lambda_fc": lam_fc,
        "deltaF": deltaF,
        "thermo_metrics": met_th,
    }


def residual_heatmap_matrix(traj_df, G, normalization):
    """Aggregate p(n,m,r)[G-1]^2 on (m,n), with a common normalization."""
    M = np.zeros((DIM, DIM), dtype=float)
    n = traj_df["n"].to_numpy(dtype=int)
    m = traj_df["m"].to_numpy(dtype=int)
    p = traj_df["prob"].to_numpy(dtype=float)
    v = p * (G - 1.0) ** 2
    for ni, mi, vi in zip(n, m, v):
        M[mi, ni] += vi
    return M / float(normalization)


def make_sm_figure_free_energy_heatmaps(traj_df, theta_th, Rmr, RJ, thermo_metrics):
    """Direct view of how the thermodynamic control suppresses Jarzynski tails."""
    YJ = thermo_metrics["YJ"]
    Gth = thermo_metrics["G"]
    p = traj_df["prob"].to_numpy(dtype=float)
    varJ = float(np.dot(p, (YJ - 1.0) ** 2))
    MJ = residual_heatmap_matrix(traj_df, YJ, varJ)
    MT = residual_heatmap_matrix(traj_df, Gth, varJ)

    positives = np.r_[MJ[MJ > 0], MT[MT > 0]]
    vmax = float(np.max(positives))
    vmin = max(float(np.min(positives)), vmax * 1e-8)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(6.0, 3.0))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.055), wspace=0.10)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])
    axes = [ax_left, ax_right]

    ims = []
    for ax, data, lab in zip(
        axes, [MJ, MT], ["(a)", "(b)"]
    ):
        masked = np.ma.masked_less_equal(data, 0.0)
        im = ax.imshow(masked, origin="lower", aspect="auto", interpolation="nearest",
                       cmap=HEATMAP_CMAP, norm=norm, rasterized=True)
        ims.append(im)
        panel_label(ax, lab)
        ax.set_xlabel(r"initial energy sector $n$")
        ticks = np.arange(0, DIM, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(-0.5, DIM - 0.5)
        ax.set_ylim(-0.5, DIM - 0.5)

    ax_left.set_ylabel(r"final energy sector $m$")
    ax_right.set_ylabel("")
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)

    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label(r"variance contribution $/\,\mathrm{Var}_{J}$", fontsize=FONT_SMALL + 0.4)
    cbar.ax.tick_params(labelsize=FONT_SMALL - 0.2, width=0.7)

    axes[0].set_title("direct Jarzynski", fontsize=FONT_SMALL + 0.5)
    axes[1].set_title("thermodynamic control", fontsize=FONT_SMALL + 0.5)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.18, top=0.92)
    save_figure(fig, "FigS_DeltaF_variance_heatmaps")


def make_sm_figure_free_energy_frontier_details(frontier):
    """Detailed displaced-reference path for the new common-target frontier."""
    fig, axes = plt.subplots(1, 3, figsize=(PRL_DOUBLE_COLUMN_WIDTH, 2.75),
                             gridspec_kw={"wspace": 0.42})

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.plot(frontier["D_achieved"], 100.0 * frontier["saving_vs_thermo"],
            color=C_BLUE, marker="o", ms=3.4)
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("saving vs thermo. control $(\%)$")

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.plot(frontier["D_achieved"], frontier["beta"], color=C_BLACK,
            marker="o", ms=2.8, label=r"$\beta J$")
    cols = [C_BLUE, C_ORANGE, C_GREEN, C_MAGENTA]
    for k, c in zip(range(4), cols):
        ax.plot(frontier["D_achieved"], frontier[f"a{k}"], color=c,
                marker="o", ms=2.4, label=fr"$\alpha_{k}$")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("optimized reference parameter")
    ax.legend(ncol=2, fontsize=FONT_SMALL - 0.3)

    ax = axes[2]
    panel_label(ax, "(c)")
    ax.plot(frontier["D_achieved"], frontier["lambda"], color=C_ORANGE,
            marker="s", ms=3.2, label=r"$\lambda$")
    ax2 = ax.twinx()
    ax2.plot(frontier["D_achieved"], frontier["correlation"], color=C_BLUE,
             marker="o", ms=3.0, label="correlation")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"control coefficient $\lambda$")
    ax2.set_ylabel("weight correlation")

    # Combined legend for twin axis.
    lines = ax.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best", fontsize=FONT_SMALL - 0.3)

    fig.subplots_adjust(left=0.075, right=0.91, bottom=0.23, top=0.96)
    save_figure(fig, "FigS_DeltaF_frontier_details")


# =============================================================================
# 11. REVISED TABLE AND SUMMARY
# =============================================================================


def write_free_energy_sampling_table(comparison, frontier, geom, analysis):
    comparison.to_csv(OUT_DIR / "Table_DeltaF_sampling_comparison.csv", index=False)

    tex = [
        r"\begin{table*}[t]",
        r"\caption{Finite-confidence sampling comparison for the same Jarzynski free-energy target. The thermodynamic control uses the endpoint-matching exact relation. The last two rows allow reference displacement up to $\delta=0.01$ and optimize, respectively, the estimator variance and the full two-sided Cram\'er--Chernoff benchmark.}",
        r"\label{tab:DeltaF-sampling}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{lrrrrr}",
        r"Estimator & $\delta$ & $\lambda$ & Var & $N_{1\%,99\%}$ & saving vs J. \\",
    ]
    row_end = r"\\"
    for _, row in comparison.iterrows():
        d = "--" if not np.isfinite(row["delta"]) else f"{row['delta']:.5g}"
        reference = str(row["reference"]).replace("_", r"\_")
        tex.append(
            f"{reference} & {d} & {row['lambda']:.6g} & "
            f"{row['variance']:.6g} & {row['N']:.6g} & "
            f"{100.0*row['saving_vs_Jarzynski']:.2f}\\% {row_end}"
        )
    tex += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}"]
    # pandas renames the 'lambda' field to lambda_ in itertuples.
    (OUT_DIR / "Table_DeltaF_sampling_comparison.tex").write_text("\n".join(tex), encoding="utf-8")

    pd.DataFrame([
        {"quantity": "DeltaF_over_J", "value": analysis["deltaF"]},
        {"quantity": "local_sqrt_coefficient", "value": geom["C_sqrt"]},
        {"quantity": "N_thermodynamic_control", "value": analysis["N_th"]},
        {"quantity": "N_Jarzynski", "value": analysis["N_J"]},
        {"quantity": "saving_thermo_vs_Jarzynski", "value": 1.0 - analysis["N_th"] / analysis["N_J"]},
        {"quantity": "frontier_N_at_delta_0.01", "value": analysis["N_fc"]},
        {"quantity": "frontier_saving_at_delta_0.01_vs_Jarzynski", "value": 1.0 - analysis["N_fc"] / analysis["N_J"]},
    ]).to_csv(OUT_DIR / "DeltaF_key_numbers.csv", index=False)


def write_revised_summary(endpoint, traj_df, frontier, geom, analysis, D_J):
    # Pandas may expose the underlying Series buffer as a read-only NumPy view
    # (notably with Copy-on-Write enabled).  Do not normalize that view in place.
    p = traj_df["prob"].to_numpy(dtype=float, copy=True)
    p = p / p.sum()
    lines = [
        "REVISED PRL FREE-ENERGY SAMPLING SUMMARY",
        "=" * 58,
        f"DIM={DIM}, L={L}, Q={Q}, U/J={UQ}, Lambda0/J={LAMBDA0}, Jtau={TAU}",
        f"E_B/J={E_B:.12f}",
        f"E_C-/J={endpoint['E']:.12f}",
        f"D_endpoint={endpoint['D']:.12f}",
        f"Delta_meas={endpoint['Delta_meas']:.12f}",
        f"Q_nc={endpoint['Qnc']:.12f}",
        f"mean_sigma={endpoint['mean_sigma']:.12f}",
        "",
        "COMMON PHYSICAL TARGET",
        f"DeltaF/J={analysis['deltaF']:.12f}",
        f"N_direct_Jarzynski={analysis['N_J']:.9g}",
        f"N_thermodynamic_control={analysis['N_th']:.9g}",
        f"saving_thermo_vs_Jarzynski={100*(1-analysis['N_th']/analysis['N_J']):.4f}%",
        f"lambda_thermodynamic={analysis['lambda_th']:.12f}",
        f"corr(Jarzynski,thermodynamic_weight)={analysis['thermo_metrics']['correlation']:.12f}",
        "",
        f"delta_budget_main={CONTROL_DELTA_BUDGET:.6g}",
        f"N_variance_opt_at_budget={analysis['N_var']:.9g}",
        f"N_finite_conf_opt_at_budget={analysis['N_fc']:.9g}",
        f"saving_finite_conf_vs_Jarzynski={100*(1-analysis['N_fc']/analysis['N_J']):.4f}%",
        f"local_sqrt_coefficient={geom['C_sqrt']:.12f}",
        "",
        f"D(theta_J || reference family from thermodynamic anchor)={D_J:.12f}",
        f"frontier_max_delta={float(frontier['delta_budget'].max()):.6g}",
        "The frontier is deliberately kept local and does not use the Jarzynski member as its own control.",
    ]
    (OUT_DIR / "generation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 12. CORE DATA: PRECOMPUTED WHEN AVAILABLE, OTHERWISE RECOMPUTE
# =============================================================================


def endpoint_from_time_row(row):
    return {
        "E": float(row["E"]),
        "p": np.array([float(row[f"p{r}"]) for r in range(Q + 1)]),
        "pseq": np.array([float(row[f"pseq{r}"]) for r in range(Q + 1)]),
        "D": float(row["D"]),
        "Delta_meas": float(row["Delta_meas"]),
        "Qnc": float(row["Qnc"]),
        "mean_sigma": float(row["mean_sigma"]),
        "beta": float(row["beta_star"]),
        "alpha": np.array([float(row[f"alpha{r}"]) for r in range(Q + 1)]),
        "maxent_resid": float(row.get("maxent_resid", np.nan)),
    }


def _find_optional_csv(name):
    candidates = [DATA_DIR / name, HERE / name, Path("/mnt/data") / name]
    for pth in candidates:
        if pth.exists():
            return pth
    return None


def load_or_recompute_nominal_core():
    time_path = _find_optional_csv("nominal_time_resolved_data.csv")
    traj_path = _find_optional_csv("endpoint_targetB_trajectory_distribution.csv")

    if USE_PRECOMPUTED_CORE and time_path is not None and traj_path is not None:
        print("Loading frozen nominal endpoint/time/trajectory data...")
        time_df = pd.read_csv(time_path)
        traj_df = pd.read_csv(traj_path)
        endpoint = endpoint_from_time_row(time_df.sort_values("Jt").iloc[-1])
        # R_mr is determined only by H_f and the record projectors.
        Rmr = np.array([
            [np.real(V_F[:, m].conj().T @ P @ V_F[:, m]) for P in P_R]
            for m in range(DIM)
        ])
        return time_df, traj_df, endpoint, Rmr, None

    print("Recomputing nominal Bose-Hubbard protocol from the Hamiltonian...")
    sol = propagate_nominal_dense()
    time_df = compute_time_resolved_nominal(sol)

    U_end = sol.sol(TAU).reshape((DIM, DIM))
    U_end, _ = polar(U_end)
    rho_f = U_end @ RHO_I @ U_end.conj().T
    E_f = float(np.real(np.trace(H_F @ rho_f)))
    p_f = np.array([np.real(np.trace(P @ rho_f)) for P in P_R])
    last = time_df.iloc[-1]
    x0 = np.r_[last["beta_star"], [last[f"alpha{r}"] for r in range(4)]]
    theta, _, logZf, alpha, resid = fit_maxent(H_F, E_f, p_f, x0)
    endpoint = endpoint_quantities(rho_f, H_F, theta, logZf, alpha)
    endpoint["maxent_resid"] = resid
    traj_df, _, Rmr = endpoint_trajectory_ensemble(U_end, endpoint)
    return time_df, traj_df, endpoint, Rmr, U_end


def maybe_generate_legacy_validation_outputs(time_df, endpoint, U_end=None):
    """Keep non-obsolete validation/SM outputs when archived CSVs are present.

    Old sampling-frontier CSVs are intentionally NOT used: they optimized the
    previous different-target objective and are not valid for the revised paper.
    """
    names = {
        "arb_fullrank": "arbitrary_initial_state_fullrank_validation.csv",
        "arb_closure": "arbitrary_initial_state_closure_and_coherence_tests.csv",
        "arb_support": "arbitrary_initial_state_support_deficit_validation.csv",
        "arb_metrology": "arbitrary_initial_state_calibration_metrology.csv",
        "compatible_detail": "compatible_pair_detail.csv",
        "compatible_final_record": "compatible_pair_final_record.csv",
    }
    data = {}
    for key, name in names.items():
        pth = _find_optional_csv(name)
        if pth is not None:
            data[key] = pd.read_csv(pth)

    # Always keep the core endpoint/time figures and table.
    make_sm_figure_time_records(time_df)
    write_table_s1(endpoint)

    if "arb_fullrank" in data:
        write_table_s2(data["arb_fullrank"], time_df)
    if all(k in data for k in ["arb_fullrank", "arb_closure", "arb_support"]):
        write_table_s3(data["arb_fullrank"], data["arb_closure"], data["arb_support"])
    if "arb_support" in data:
        make_sm_figure_support_deficit(data["arb_support"])
    if "arb_metrology" in data:
        make_sm_figure_metrology(data["arb_metrology"])
    if all(k in data for k in ["compatible_detail", "compatible_final_record"]):
        make_sm_figure_closure_precomputed(data["compatible_detail"], data["compatible_final_record"])
    elif U_end is not None:
        make_sm_figure_closure(U_end)


# =============================================================================
# 13. MAIN
# =============================================================================


def main():
    set_prl_style()
    print("Output folder:", OUT_DIR)

    time_df, traj_df, endpoint, Rmr, U_end = load_or_recompute_nominal_core()
    theta_th = thermodynamic_theta(endpoint)
    RJ, deltaF, _ = final_canonical_free_energy()

    print("Computing common-target free-energy control-variate frontier...")
    frontier, geom, D_J = build_free_energy_frontier(theta_th, traj_df, Rmr, RJ)

    print("Generating revised main-text figures...")
    make_main_figure_1_updated(time_df, endpoint)
    comparison, analysis = make_main_figure_2_free_energy(
        frontier, theta_th, geom, traj_df, Rmr, RJ, deltaF
    )

    print("Generating revised Supplemental-Material figures...")
    make_sm_figure_free_energy_heatmaps(
        traj_df, theta_th, Rmr, RJ, analysis["thermo_metrics"]
    )
    make_sm_figure_free_energy_frontier_details(frontier)
    # The original theorem-level rare-event diagnostic is still useful and is
    # not part of the obsolete different-target sampling comparison.
    make_sm_figure_rare_events(traj_df)

    print("Writing revised sampling table and summary...")
    write_free_energy_sampling_table(comparison, frontier, geom, analysis)
    write_revised_summary(endpoint, traj_df, frontier, geom, analysis, D_J)

    print("Generating any available non-obsolete legacy validation outputs...")
    maybe_generate_legacy_validation_outputs(time_df, endpoint, U_end=U_end)

    print("\nDone.")
    print(f"Output folder: {OUT_DIR}")
    print(f"  DeltaF/J                  = {deltaF:.12f}")
    print(f"  N direct Jarzynski        = {analysis['N_J']:.6e}")
    print(f"  N thermodynamic control   = {analysis['N_th']:.6e}")
    print(f"  N finite-conf opt d=0.01  = {analysis['N_fc']:.6e}")
    print(f"  local sqrt coefficient    = {geom['C_sqrt']:.6f}")


if __name__ == "__main__":
    main()
