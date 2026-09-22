#!/usr/bin/env python3
"""
Standalone validation for the arbitrary-initial-state Target-B calculations.

Model:
  L=4, Q=4 fixed-particle Bose-Hubbard chain
  J=1, U_Q/J=(1.3/3), Lambda_0/J=20, J tau=20
  canonical reference beta_i J=0.2
  linear ramp Lambda(t)=Lambda_0(1-t/tau)

Outputs:
  arbitrary_initial_state_fullrank_validation.csv
  arbitrary_initial_state_support_deficit_validation.csv
  arbitrary_initial_state_closure_and_coherence_tests.csv
  arbitrary_initial_state_calibration_metrology.csv

Independent propagators:
  DOP853 + polar unitarization
  fourth-order two-node Gauss-Magnus, J dt=0.01
"""

import csv
import itertools
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm, norm, null_space, polar
from scipy.optimize import least_squares, linprog

OUT = Path(__file__).resolve().parent

L = 4
Q = 4
J = 1.0
UQ = 1.3 / 3.0
LAMBDA0 = 20.0
TAU = 20.0
BETA_I = 0.2

basis = [occ for occ in itertools.product(range(Q + 1), repeat=L) if sum(occ) == Q]
d = len(basis)
index = {b: i for i, b in enumerate(basis)}


def bh_ham(Lam):
    H = np.zeros((d, d), dtype=complex)
    for i, b in enumerate(basis):
        n = np.asarray(b)
        H[i, i] += UQ / 2.0 * np.sum(n * (n - 1)) + Lam * np.sum(n[L // 2 :])
    for i, b0 in enumerate(basis):
        b = list(b0)
        for j in range(L - 1):
            if b[j + 1] > 0:
                nb = b.copy()
                amp = math.sqrt((nb[j] + 1) * nb[j + 1])
                nb[j] += 1
                nb[j + 1] -= 1
                H[index[tuple(nb)], i] += -J * amp
            if b[j] > 0:
                nb = b.copy()
                amp = math.sqrt((nb[j + 1] + 1) * nb[j])
                nb[j + 1] += 1
                nb[j] -= 1
                H[index[tuple(nb)], i] += -J * amp
    return (H + H.conj().T) / 2.0


Pr = []
for r in range(Q + 1):
    diag = np.array([1.0 if sum(b[L // 2 :]) == r else 0.0 for b in basis])
    Pr.append(np.diag(diag))

NR = sum(r * Pr[r] for r in range(Q + 1))
Hi = bh_ham(LAMBDA0)
Hf = bh_ham(0.0)
ei, Vi = eigh(Hi)
ef, Vf = eigh(Hf)


def lambda_t(t):
    return LAMBDA0 * (1.0 - t / TAU)


def propagate_dop853():
    def rhs(t, y):
        U = y.reshape((d, d))
        H = Hf + lambda_t(t) * NR
        return (-1j * H @ U).reshape(-1)

    y0 = np.eye(d, dtype=complex).reshape(-1)
    sol = solve_ivp(
        rhs,
        (0.0, TAU),
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    U = sol.y[:, -1].reshape((d, d))
    Uu, _ = polar(U)
    return Uu


def propagate_magnus4(dt=0.01):
    steps = int(round(TAU / dt))
    dt = TAU / steps
    c = math.sqrt(3.0) / 6.0
    U = np.eye(d, dtype=complex)
    for k in range(steps):
        t0 = k * dt
        t1 = t0 + (0.5 - c) * dt
        t2 = t0 + (0.5 + c) * dt
        H1 = Hf + lambda_t(t1) * NR
        H2 = Hf + lambda_t(t2) * NR
        comm = H1 @ H2 - H2 @ H1
        Omega = -1j * dt * 0.5 * (H1 + H2) + math.sqrt(3.0) * dt * dt * comm / 12.0
        U = expm(Omega) @ U
    return U


def gibbs_probs(beta):
    x = -beta * ei
    x -= np.max(x)
    p = np.exp(x)
    return p / p.sum()


q = gibbs_probs(BETA_I)


def expm_herm_neg(K):
    vals, vecs = eigh(K)
    x = -vals
    xmax = np.max(x)
    w = np.exp(x - xmax)
    rho = (vecs * w) @ vecs.conj().T
    rho /= w.sum()
    logZ = xmax + math.log(w.sum())
    return rho, logZ


def maxent_constraints(theta):
    beta = theta[0]
    alphas = np.r_[theta[1:], 0.0]
    A = sum(a * P for a, P in zip(alphas, Pr))
    rho, logZ = expm_herm_neg(beta * Hf + A)
    c = np.r_[
        np.real(np.trace(Hf @ rho)),
        [np.real(np.trace(P @ rho)) for P in Pr[:4]],
    ]
    return c, rho, logZ, alphas


def fit_maxent(E_target, pR_target, x0):
    target = np.r_[E_target, pR_target[:4]]

    def fun(x):
        return maxent_constraints(x)[0] - target

    res = least_squares(
        fun,
        x0,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=5000,
        x_scale="jac",
    )
    c, rho, logZ, alphas = maxent_constraints(res.x)
    return res, rho, logZ, alphas, c - target


def entropy_from_probs(p):
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))


def endpoint_analysis(p_init, Umat, fit_x0):
    rhoi = Vi @ np.diag(p_init) @ Vi.conj().T
    rhof = Umat @ rhoi @ Umat.conj().T
    Ei = float(np.dot(p_init, ei))
    Ef = float(np.real(np.trace(Hf @ rhof)))
    pR = np.array([np.real(np.trace(P @ rhof)) for P in Pr])

    res, rhobar, logZ, alphas, resid = fit_maxent(Ef, pR, fit_x0)
    beta = float(res.x[0])
    A = sum(a * P for a, P in zip(alphas, Pr))

    eHb = (Vf * np.exp(-beta * ef)) @ Vf.conj().T
    eA = sum(np.exp(-a) * P for a, P in zip(alphas, Pr))
    Zseq = float(np.real(np.trace(eHb @ eA)))
    Qnc = math.log(Zseq) - logZ

    diagf = np.real(np.diag(Vf.conj().T @ rhof @ Vf))
    rhof_d = (Vf * diagf) @ Vf.conj().T
    pseq = np.array([np.real(np.trace(P @ rhof_d)) for P in Pr])
    Delta_meas = float(np.dot(alphas, pseq - pR))

    S = entropy_from_probs(p_init)
    Drel = -S + beta * Ef + float(np.dot(alphas, pR)) + logZ
    mean_sigma = Drel + Delta_meas + Qnc

    Uem = Vf.conj().T @ Umat @ Vi
    T = np.abs(Uem) ** 2
    Rmr = np.array(
        [
            [np.real(Vf[:, m].conj().T @ P @ Vf[:, m]) for P in Pr]
            for m in range(d)
        ]
    )

    Psi_f = -logZ
    ift = 0.0
    mean_direct = 0.0
    for n in range(d):
        if p_init[n] <= 0:
            continue
        for m in range(d):
            if T[m, n] < 1e-18:
                continue
            for r in range(Q + 1):
                prob = p_init[n] * T[m, n] * Rmr[m, r]
                if prob < 1e-24:
                    continue
                sigma = (
                    beta * ef[m]
                    + alphas[r]
                    + math.log(p_init[n])
                    + Qnc
                    - Psi_f
                )
                ift += prob * math.exp(-sigma)
                mean_direct += prob * sigma

    return {
        "rhoi": rhoi,
        "rhof": rhof,
        "Ei": Ei,
        "Ef": Ef,
        "pR": pR,
        "pseq": pseq,
        "theta": res.x,
        "alphas": alphas,
        "logZ": logZ,
        "Zseq": Zseq,
        "Qnc": Qnc,
        "Delta_meas": Delta_meas,
        "D": Drel,
        "mean": mean_sigma,
        "ift": ift,
        "mean_direct": mean_direct,
        "fit_resid": float(np.max(np.abs(resid))),
        "T": T,
        "Rmr": Rmr,
    }


U_dop = propagate_dop853()
U_mag = propagate_magnus4(0.01)

x0 = np.array(
    [
        0.626142210198,
        -0.343921773387,
        -0.302435442300,
        -0.051655091588,
        0.188285336699,
    ]
)

# Three full-rank baseline preparations.
p_beta12 = gibbs_probs(0.12)
p_mix = 0.7 * q + 0.3 * gibbs_probs(0.08)
p_mix /= p_mix.sum()

# Build the deterministic same-(E,R) pair.
Rin = np.array(
    [
        [np.real(Vi[:, n].conj().T @ P @ Vi[:, n]) for n in range(d)]
        for P in Pr
    ]
)
C = np.vstack([np.ones(d), ei, Rin])
b = C @ q
c2 = ei**2
lp_max = linprog(-c2, A_eq=C, b_eq=b, bounds=[(0, None)] * d, method="highs")
lp_min = linprog(c2, A_eq=C, b_eq=b, bounds=[(0, None)] * d, method="highs")
if not (lp_max.success and lp_min.success):
    raise RuntimeError("Compatibility-fiber linear program failed.")
pA = 0.8 * q + 0.2 * lp_max.x
pB = 0.8 * q + 0.2 * lp_min.x

states = {
    "canonical": q,
    "gibbs_beta0.12": p_beta12,
    "nonGibbs_mixture": p_mix,
    "compatible_A": pA,
    "compatible_B": pB,
}

full_rows = []
analyses = {}
for name, p in states.items():
    a = endpoint_analysis(p, U_dop, x0)
    bmag = endpoint_analysis(p, U_mag, a["theta"])
    analyses[name] = a
    full_rows.append(
        {
            "state": name,
            "Ei_DOP853": a["Ei"],
            "Ef_DOP853": a["Ef"],
            "D_DOP853": a["D"],
            "Delta_meas_DOP853": a["Delta_meas"],
            "Qnc_DOP853": a["Qnc"],
            "mean_sigma_DOP853": a["mean"],
            "IFT_DOP853": a["ift"],
            "fit_resid_DOP853": a["fit_resid"],
            "Ef_Magnus_dt0.01": bmag["Ef"],
            "D_Magnus_dt0.01": bmag["D"],
            "Delta_meas_Magnus_dt0.01": bmag["Delta_meas"],
            "Qnc_Magnus_dt0.01": bmag["Qnc"],
            "mean_sigma_Magnus_dt0.01": bmag["mean"],
            "IFT_Magnus_dt0.01": bmag["ift"],
            "fit_resid_Magnus_dt0.01": bmag["fit_resid"],
            "abs_Ef_solver_diff": abs(a["Ef"] - bmag["Ef"]),
            "abs_D_solver_diff": abs(a["D"] - bmag["D"]),
            "abs_Qnc_solver_diff": abs(a["Qnc"] - bmag["Qnc"]),
        }
    )

with open(OUT / "arbitrary_initial_state_fullrank_validation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=full_rows[0].keys())
    w.writeheader()
    w.writerows(full_rows)


def rankdef_analysis(K, Umat, fit_x0):
    p = np.zeros(d)
    p[:K] = q[:K]
    p /= p.sum()
    rhoi = Vi @ np.diag(p) @ Vi.conj().T
    rhof = Umat @ rhoi @ Umat.conj().T
    Ef0 = float(np.real(np.trace(Hf @ rhof)))
    pR0 = np.array([np.real(np.trace(P @ rhof)) for P in Pr])
    res, _, logZ, alphas, resid = fit_maxent(Ef0, pR0, fit_x0)
    beta = float(res.x[0])
    A = sum(a * P for a, P in zip(alphas, Pr))
    eHb = (Vf * np.exp(-beta * ef)) @ Vf.conj().T
    eA = sum(np.exp(-a) * P for a, P in zip(alphas, Pr))
    Zseq = float(np.real(np.trace(eHb @ eA)))
    Qnc = math.log(Zseq) - logZ
    Psi_f = -logZ

    Uem = Vf.conj().T @ Umat @ Vi
    T = np.abs(Uem) ** 2
    Rmr = np.array(
        [
            [np.real(Vf[:, m].conj().T @ P @ Vf[:, m]) for P in Pr]
            for m in range(d)
        ]
    )

    direct = 0.0
    for n in range(K):
        for m in range(d):
            for r in range(Q + 1):
                prob = p[n] * T[m, n] * Rmr[m, r]
                if prob < 1e-25:
                    continue
                sigma = (
                    beta * ef[m]
                    + alphas[r]
                    + math.log(p[n])
                    + Qnc
                    - Psi_f
                )
                direct += prob * math.exp(-sigma)

    Bseq = np.zeros((d, d), dtype=complex)
    for m in range(d):
        Pm = np.outer(Vf[:, m], Vf[:, m].conj())
        Bseq += math.exp(-beta * ef[m]) * (Pm @ eA @ Pm)

    Si = Vi[:, :K] @ Vi[:, :K].conj().T
    operator = float(np.real(np.trace(Bseq @ Umat @ Si @ Umat.conj().T)) / Zseq)
    return direct, operator, 1.0 - operator, res.x, float(np.max(np.abs(resid)))


supp_rows = []
for K in [5, 10, 20, 30, 34]:
    a = rankdef_analysis(K, U_dop, x0)
    bmag = rankdef_analysis(K, U_mag, a[3])
    supp_rows.append(
        {
            "K_supported": K,
            "IFT_direct_DOP853": a[0],
            "IFT_operator_DOP853": a[1],
            "Lambda_supp_DOP853": a[2],
            "IFT_direct_Magnus_dt0.01": bmag[0],
            "IFT_operator_Magnus_dt0.01": bmag[1],
            "Lambda_supp_Magnus_dt0.01": bmag[2],
            "abs_solver_diff_IFT": abs(a[0] - bmag[0]),
        }
    )

with open(
    OUT / "arbitrary_initial_state_support_deficit_validation.csv", "w", newline=""
) as f:
    w = csv.DictWriter(f, fieldnames=supp_rows[0].keys())
    w.writeheader()
    w.writerows(supp_rows)


def trace_distance(rho, sigma):
    vals = np.linalg.eigvalsh((rho - sigma + (rho - sigma).conj().T) / 2.0)
    return 0.5 * float(np.sum(np.abs(vals)))


# Coherent state with exactly canonical energy populations.
phi = 0.2 * np.arange(d) ** 2
psi = Vi @ (np.sqrt(q) * np.exp(1j * phi))
psi /= np.linalg.norm(psi)
rho_coh = np.outer(psi, psi.conj())
rho_can = Vi @ np.diag(q) @ Vi.conj().T
rho_coh_f = U_dop @ rho_coh @ U_dop.conj().T
rho_can_f = U_dop @ rho_can @ U_dop.conj().T
pR_coh = np.array([np.real(np.trace(P @ rho_coh_f)) for P in Pr])
pR_can = np.array([np.real(np.trace(P @ rho_can_f)) for P in Pr])

closure_rows = [
    {
        "test": "coherent_canonical_populations",
        "metric": "trace_distance_unmeasured_vs_TPM_endpoint",
        "value": trace_distance(rho_coh, rho_can),
    },
    {
        "test": "coherent_canonical_populations",
        "metric": "final_record_TV_unmeasured_vs_TPM",
        "value": 0.5 * float(np.sum(np.abs(pR_coh - pR_can))),
    },
    {
        "test": "compatible_pair",
        "metric": "construction_extreme_fraction",
        "value": 0.2,
    },
    {
        "test": "compatible_pair",
        "metric": "initial_energy_distribution_TV",
        "value": 0.5 * float(np.sum(np.abs(pA - pB))),
    },
    {
        "test": "compatible_pair",
        "metric": "Gamma_operator_sup_difference",
        "value": float(np.max(np.abs(np.log(pA / q) - np.log(pB / q)))),
    },
    {
        "test": "compatible_pair",
        "metric": "final_energy_difference_J",
        "value": abs(analyses["compatible_A"]["Ef"] - analyses["compatible_B"]["Ef"]),
    },
    {
        "test": "compatible_pair",
        "metric": "final_record_TV",
        "value": 0.5
        * float(
            np.sum(
                np.abs(
                    analyses["compatible_A"]["pR"] - analyses["compatible_B"]["pR"]
                )
            )
        ),
    },
    {
        "test": "model",
        "metric": "constraint_matrix_rank",
        "value": int(np.linalg.matrix_rank(C, tol=1e-10)),
    },
    {
        "test": "model",
        "metric": "compatible_energy_population_fiber_dimension",
        "value": d - int(np.linalg.matrix_rank(C, tol=1e-10)),
    },
    {
        "test": "model",
        "metric": "Hi_min_gap_J",
        "value": float(np.min(np.diff(ei))),
    },
]

with open(
    OUT / "arbitrary_initial_state_closure_and_coherence_tests.csv", "w", newline=""
) as f:
    w = csv.DictWriter(f, fieldnames=["test", "metric", "value"])
    w.writeheader()
    w.writerows(closure_rows)


def b_distribution(analysis):
    beta = analysis["theta"][0]
    alphas = analysis["alphas"]
    Zseq = analysis["Zseq"]
    weights_m = (
        np.array(
            [
                np.sum(analysis["Rmr"][m, :] * np.exp(-alphas))
                for m in range(d)
            ]
        )
        * np.exp(-beta * ef)
    )
    return (weights_m[:, None] * analysis["T"]).sum(axis=0) / Zseq


def N_see_one(pmin, alpha=0.01):
    return math.ceil(math.log(alpha) / math.log1p(-pmin))


def N_all_union(pmin, K=35, alpha=0.01):
    return math.ceil(math.log(alpha / K) / math.log1p(-pmin))


def N_uniform_gamma(pmin, delta_gamma=0.1, alpha=0.01, K=35):
    eta = 1.0 - math.exp(-delta_gamma)
    return math.ceil(3.0 * math.log(2.0 * K / alpha) / (pmin * eta * eta))


met_cases = [
    ("gibbs_beta0.12", p_beta12, analyses["gibbs_beta0.12"]),
    ("nonGibbs_mixture", p_mix, analyses["nonGibbs_mixture"]),
    ("canonical_like_if_unknown", q, analyses["canonical"]),
]
met_rows = []
for name, p, a in met_cases:
    bn = b_distribution(a)
    coeff = math.sqrt(float(np.sum(bn**2 / p) - 1.0))
    bias_coeff = float(np.sum(bn * (1.0 - p) / p))
    met_rows.append(
        {
            "state": name,
            "p_min": float(np.min(p)),
            "N_99pct_rarest_seen_once": N_see_one(float(np.min(p))),
            "N_99pct_all_35_seen_union_bound": N_all_union(float(np.min(p))),
            "delta_method_IFT_SE_coefficient": coeff,
            "N_for_1pct_one_SE_delta": math.ceil((coeff / 0.01) ** 2),
            "N_for_1pct_99pct_Gaussian_delta": math.ceil(
                (2.575829 * coeff / 0.01) ** 2
            ),
            "second_order_bias_coefficient": bias_coeff,
            "N_uniform_abs_gamma_le_0.1_99pct_Chernoff": N_uniform_gamma(
                float(np.min(p))
            ),
        }
    )

with open(
    OUT / "arbitrary_initial_state_calibration_metrology.csv", "w", newline=""
) as f:
    w = csv.DictWriter(f, fieldnames=met_rows[0].keys())
    w.writeheader()
    w.writerows(met_rows)

print("Finished.")
print("DOP853/Magnus operator-norm difference:", norm(U_dop - U_mag, 2))
for row in full_rows:
    print(
        row["state"],
        "IFT(DOP853)=", row["IFT_DOP853"],
        "IFT(Magnus)=", row["IFT_Magnus_dt0.01"],
    )
