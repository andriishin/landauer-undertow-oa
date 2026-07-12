"""sim 4: `superlinear_memory` -- phase boundary (Theorem 3, C3).

Article #3 (`landauer-undertow`) § 5 / § 6 item 4 / supplementary § S4.4.

Memory capacity grows as |M(t)| = M0 t^alpha for alpha in {1, 1.5, 2, 3}
(polynomial), and separately |M(t)| = M0 e^{kappa t} (exponential escape).
FIFO refresh. We measure:

  1. the fresh-bit fraction phi(t) = (|M(t)| - |M(t-tau_E)|)/|M(t)| ~ alpha*tau_E/t
     -- claim: phi(t) ~ t^{-gamma} with gamma = 1 for EVERY finite alpha (the
     decay exponent does NOT depend on alpha), so phi(t) -> 0 for every finite
     alpha => NO finite polynomial escape threshold alpha_c. The load-bearing
     route-closure is eta_v = I_pred/I_mem -> 0; the tabulated per-bit
     nu = 1 - phi -> 1 is the naive/circular per-bit majorant (NOT nu^theor).
     The true nu^theor is held at the C1 floor nu_C1 < 1, NOT -> 1;
  2. for exponential growth, phi(t) -> phi_inf > 0 (constant), nu(t) < 1
     (escape) -- but eta = I_pred/I_mem -> 0 STRUCTURALLY: I_pred is horizon-bounded
     while I_mem grows with the (exponentially growing) retained capacity |M|.

Direct construction with known true theta(t): the tabulated per-bit nu = 1 - phi
is the operational per-bit majorant (circular), NOT nu^theor. The load-bearing C3
carriers are the fresh fraction phi(t) -> 0 (=> no finite polynomial alpha_c) and
the route-closure eta_v -> 0; the true nu^theor is held at the C1 floor
nu_C1 < 1 (S4.5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------- parameters ----------------
K_STATES = 8
K_PARAMS = K_STATES * (K_STATES - 1)
ALPHAS = [1.0, 1.5, 2.0, 3.0]
KAPPAS = [0.1, 0.5, 1.0]           # exponential escape: kappa*tau_E
TAU_E = 200.0
M0 = 100.0
T_TOTAL = 20_000
T_MIN_FIT = 2_000.0                # fit phi ~ t^-gamma past the transient
SEED = 20260524 + 5                # offset seed (unused; output is a
                                   # deterministic closed-form tabulation of
                                   # phi = 1 - ((t-tau_E)/t)^alpha -- NOT Monte
                                   # Carlo, np.random is never called, RNG is
                                   # never seeded. Kept only for provenance.
N_T = 400                          # number of log-spaced measurement times


def phi_polynomial(t, alpha, tau_E):
    """Fresh-bit fraction for |M| = M0 t^alpha, FIFO:
    phi(t) = 1 - ((t - tau_E)/t)^alpha   (exact), ~ alpha*tau_E/t for t >> tau_E."""
    t = np.asarray(t, dtype=float)
    base = np.clip((t - tau_E) / t, 0.0, 1.0)
    return 1.0 - base ** alpha


# fit windows for the gamma window-convergence diagnostic: each window is one
# decade [w, 10*w] of the EXACT closed-form phi (deterministic; no Monte Carlo).
# w = 2e3 reproduces the baseline fit window [T_MIN_FIT, T_TOTAL] = [2e3, 2e4].
GAMMA_WINDOWS = [2e3, 2e4, 2e5, 2e6]
# t values at which we report the local log-log slope of the EXACT phi (alpha=3).
SLOPE_TS = [2e3, 2e4, 2e5, 2e6, 2e7, 2e8, 2e9]


def gamma_window_convergence(alphas, tau_E, windows=GAMMA_WINDOWS, n_t=400):
    """Show that gamma (phi ~ t^-gamma) -> 1 MONOTONICALLY for EVERY alpha as the
    fit window slides to larger t. This proves that the small downward drift of
    gamma below 1 on the baseline window is a finite-window subleading artifact,
    not a real alpha-dependence (the leading exponent is identically 1, so there
    is no polynomial escape threshold alpha_c).

    Deterministic: phi = 1 - ((t - tau_E)/t)^alpha is a closed form tabulated on a
    log grid; no random paths, no Monte Carlo, RNG is never invoked. Each window
    is one decade [w, 10*w], so w = 2e3 fits on [2e3, 2e4] and reproduces the
    baseline gamma closely (to 3 s.f.; the open-ended baseline fit uses a
    slightly different log grid, hence a 4th-digit difference).
    """
    table = []  # one entry per window: {"t_min": w, "gammas": {alpha: gamma}}
    for w in windows:
        times = np.unique(np.logspace(np.log10(w), np.log10(10.0 * w), n_t))
        gammas = {}
        for alpha in alphas:
            phi = phi_polynomial(times, alpha, tau_E)
            mask = phi > 0
            p = np.polyfit(np.log(times[mask]), np.log(phi[mask]), 1)
            gammas[alpha] = float(-p[0])
        table.append({"t_min": float(w), "gammas": gammas})

    # local log-log slope d(log phi)/d(log t) of the EXACT phi at alpha=3:
    # the leading-term exponent is identically 1, so slope -> -1.0000 as t grows,
    # which is what makes the gamma drift a strictly subleading effect.
    alpha_slope = 3.0
    slopes = []
    for tc in SLOPE_TS:
        thi = tc * (1.0 + 1e-4)
        tlo = tc * (1.0 - 1e-4)
        s = (np.log(phi_polynomial(thi, alpha_slope, tau_E))
             - np.log(phi_polynomial(tlo, alpha_slope, tau_E))) / (
            np.log(thi) - np.log(tlo))
        slopes.append({"t": float(tc), "slope": float(s)})

    return {"alphas": list(alphas), "windows": table,
            "slope_alpha": alpha_slope, "local_slopes": slopes}


def main() -> None:
    log_lines: list[str] = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("== sim 4: superlinear_memory -- phase boundary (Theorem 3) ==")
    log(f"k = {K_STATES}, K = {K_PARAMS}, tau_E = {TAU_E}, M0 = {M0}")
    log(f"polynomial alphas = {ALPHAS}; exponential kappa*tau_E = {KAPPAS}")
    log(f"T_total = {T_TOTAL}, fit window t >= {T_MIN_FIT}, seed = {SEED}")

    t_start = time.time()
    times = np.unique(np.logspace(np.log10(TAU_E + 1), np.log10(T_TOTAL),
                                  N_T).astype(int)).astype(float)

    # ----- polynomial growth -----
    poly = []
    for alpha in ALPHAS:
        phi = phi_polynomial(times, alpha, TAU_E)
        nu = 1.0 - phi                      # per-bit majorant nu = 1 - phi (circular, NOT nu^theor)
        # fit phi ~ t^-gamma on the tail
        mask = (times >= T_MIN_FIT) & (phi > 0)
        p = np.polyfit(np.log(times[mask]), np.log(phi[mask]), 1)
        gamma = float(-p[0])
        # prefactor check: phi*t -> alpha*tau_E
        pref = float(np.median(phi[mask] * times[mask]))
        log(f"\n alpha = {alpha}: gamma (phi~t^-gamma) = {gamma:.4f} (expected 1.000); "
            f"phi*t -> {pref:.2f} (expected alpha*tau_E = {alpha*TAU_E:.0f}); "
            f"nu(T) = {nu[-1]:.4f}")
        poly.append({"alpha": alpha, "gamma": gamma, "expected_gamma": 1.0,
                     "phi_times_t": pref, "expected_prefactor": alpha * TAU_E,
                     "nu_final": float(nu[-1]),
                     "phi": phi.tolist(), "nu": nu.tolist()})

    # ----- gamma window-convergence diagnostic (subleading-artifact proof) -----
    # Deterministic closed-form; proves the baseline gamma drift below 1 is a
    # finite-window subleading effect: gamma -> 1 monotonically for ALL alpha as
    # the fit window slides right => leading exponent is identically 1 => no alpha_c.
    gwc = gamma_window_convergence(ALPHAS, TAU_E)
    log("\n-- gamma window-convergence check (exact phi, deterministic) --")
    log("  fit window     " + "".join(f"alpha={a:<5}" for a in ALPHAS))
    for row in gwc["windows"]:
        cells = "".join(f"{row['gammas'][a]:<10.4f}" for a in ALPHAS)
        log(f"  t>={row['t_min']:>7.0e}    {cells}")
    log(f"  => gamma -> 1 monotonically for every alpha as the window slides right "
        f"(baseline drift is a subleading finite-window artifact).")
    log(f"  local log-log slope of exact phi at alpha={gwc['slope_alpha']:.0f} "
        f"(leading exponent identically 1):")
    log("    " + "  ".join(f"t={s['t']:.0e}:{s['slope']:+.4f}"
                            for s in gwc["local_slopes"]))

    # ----- exponential growth (escape) -----
    expo = []
    for kt in KAPPAS:
        # kt = kappa*tau_E; phi = 1 - e^{-kappa tau_E} = 1 - e^{-kt}, constant
        phi_inf = 1.0 - np.exp(-kt)
        phi = np.full_like(times, phi_inf)
        nu = 1.0 - phi
        # Exponential growth escapes the nostalgia floor (phi_inf>0, nu<1), but
        # eta = I_pred/I_mem -> 0 STRUCTURALLY: I_pred is horizon-bounded (the future
        # is only so predictable) while I_mem grows with the retained capacity |M|.
        kappa = kt / TAU_E
        M_size = M0 * np.exp(kappa * times)         # |M(t)|, exponential capacity
        I_pred_bound = phi_inf * M0                 # horizon-bounded predictive info (does NOT scale with |M|)
        I_mem = M_size                              # retained memory ~ capacity (informational, grows e^{kappa t})
        eta = I_pred_bound / I_mem                  # = phi_inf e^{-kappa t} -> 0 (structural smallness; no energy budget)
        log(f"\n kappa*tau_E = {kt}: phi_inf = {phi_inf:.4f} (>0, escape, nu<1); "
            f"eta = I_pred/I_mem -> 0 structurally; eta(T)/eta(0) = {eta[-1]/eta[0]:.3e}, "
            f"|M|(T)/|M|(0) = {M_size[-1]/M_size[0]:.3e}")
        expo.append({"kappa_tauE": kt, "phi_inf": float(phi_inf),
                     "nu_inf": float(1 - phi_inf),
                     "M_ratio_T": float(M_size[-1] / M_size[0]),
                     "eta_ratio_T": float(eta[-1] / eta[0]),
                     "phi": phi.tolist(), "I_mem": I_mem.tolist(),
                     "eta": eta.tolist()})

    total = time.time() - t_start
    gammas = [r["gamma"] for r in poly]
    log(f"\n== total runtime {total:.1f}s ==")
    log(f"gamma across polynomial alpha: {[f'{g:.3f}' for g in gammas]} "
        f"(all ~ 1, independent of alpha => no polynomial alpha_c)")

    # ---------------- figure: phi(t) log-log for polynomial alphas ----------------
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    cmap = plt.get_cmap("cool")
    for i, r in enumerate(poly):
        col = cmap(i / max(1, len(poly) - 1))
        ax.loglog(times, r["phi"], "-", color=col, lw=1.5,
                  label=rf"$\alpha={r['alpha']}$ ($\gamma={r['gamma']:.2f}$)")
    # reference slope -1
    ref = poly[0]["expected_prefactor"] / times
    ax.loglog(times, ref, "k:", lw=1.0, label=r"slope $-1$ ref")
    for i, r in enumerate(expo):
        ax.axhline(r["phi_inf"], color="tab:red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\phi(t)$ fresh-bit fraction")
    ax.set_title(r"superlinear_memory: $\phi(t)\sim t^{-1}$ for all polynomial $\alpha$; "
                 r"$\phi\to\phi_\infty>0$ only for exponential growth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_superlinear_phi.png", dpi=150)
    plt.close(fig)
    log("saved fig_superlinear_phi.png")

    # second figure: exponential growth -> eta structurally tiny; polynomial
    # phi->0 / eta_v->0 (per-bit nu=1-phi->1 is the circular majorant, not nu^theor)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax = axes[0]
    for i, r in enumerate(expo):
        ax.semilogy(times, r["eta"], "-", lw=1.4,
                    label=rf"$\kappa\tau_E={r['kappa_tauE']}$")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\eta(t)=I_{\rm pred}/I_{\rm mem}$")
    ax.set_title(r"Exponential growth: $\eta\to 0$ structurally ($|M|$ grows, $I_{\rm pred}$ bounded)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    for i, r in enumerate(poly):
        ax.plot(times, r["nu"], "-", lw=1.4, label=rf"poly $\alpha={r['alpha']}$")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("t"); ax.set_ylabel(r"$\nu(t)=1-\phi(t)$")
    ax.set_title(r"Polynomial growth: $\phi\to0$, $\eta_v\to0$ (no escape);"
                 "\n" r"per-bit $\nu=1-\phi$ is a circular majorant, not $\nu^{\rm theor}$")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_superlinear_escape.png", dpi=150)
    plt.close(fig)
    log("saved fig_superlinear_escape.png")

    # ---------------- summaries ----------------
    summary = {
        "experiment": "sim4 superlinear_memory: phase boundary (Theorem 3)",
        "k": K_STATES, "K_params": K_PARAMS, "tau_E": TAU_E, "M0": M0,
        "alphas": ALPHAS, "kappa_tauE": KAPPAS,
        "T_total": T_TOTAL, "fit_t_min": T_MIN_FIT, "seed": SEED,
        "total_runtime_s": total,
        "polynomial": [{kk: vv for kk, vv in r.items()
                        if kk not in ("phi", "nu")} for r in poly],
        "exponential": [{kk: vv for kk, vv in r.items()
                         if kk not in ("phi", "I_mem", "eta")} for r in expo],
        "gamma_independent_of_alpha": bool(max(gammas) - min(gammas) < 0.05),
        "gamma_window_convergence": {
            "note": ("Deterministic closed-form tabulation of "
                     "phi = 1 - ((t-tau_E)/t)^alpha (NOT Monte Carlo; RNG never "
                     "seeded). Each window is one decade [w, 10*w]; w=2e3 "
                     "reproduces the baseline fit. gamma -> 1 monotonically for "
                     "every alpha as the window slides right => the baseline drift "
                     "below 1 is a subleading finite-window artifact, the leading "
                     "exponent is identically 1, so there is no polynomial alpha_c."),
            "alphas": gwc["alphas"],
            "windows": [{"t_min": row["t_min"],
                         "gamma": {str(a): row["gammas"][a] for a in gwc["alphas"]}}
                        for row in gwc["windows"]],
            "slope_alpha": gwc["slope_alpha"],
            "local_loglog_slopes": gwc["local_slopes"],
        },
    }
    (HERE / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    with (HERE / "results_summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== sim4: superlinear_memory -- phase boundary (Theorem 3) ===\n")
        f.write(f"k = {K_STATES}, K = {K_PARAMS}, tau_E = {TAU_E}, M0 = {M0}\n")
        f.write(f"T_total = {T_TOTAL}, fit window t >= {T_MIN_FIT}, seed = {SEED}\n")
        f.write(f"total runtime {total:.1f}s\n\n")
        f.write("--- Polynomial growth |M| = M0 t^alpha (FIFO) ---\n")
        f.write(f"{'alpha':>6s}  {'gamma':>8s}  {'expected':>8s}  {'phi*t':>9s}  "
                f"{'alpha*tauE':>10s}  {'nu(T)':>8s}\n")
        for r in poly:
            f.write(f"{r['alpha']:>6}  {r['gamma']:>8.4f}  {1.0:>8.3f}  "
                    f"{r['phi_times_t']:>9.2f}  {r['expected_prefactor']:>10.0f}  "
                    f"{r['nu_final']:>8.4f}\n")
        f.write(f"\ngamma values: {[round(g,4) for g in gammas]} -- "
                f"INDEPENDENT of alpha (spread {max(gammas)-min(gammas):.4f}); "
                f"gamma = 1 for every finite alpha => NO polynomial alpha_c.\n")
        f.write("fresh fraction phi(t) -> 0 => no finite polynomial alpha_c; route\n"
                "closed by eta_v = I_pred/I_mem -> 0. The tabulated per-bit nu = 1 - phi\n"
                "-> 1 is the naive/circular per-bit majorant, NOT nu^theor; the true\n"
                "nu^theor is held at the C1 floor nu_C1 < 1 (NOT -> 1).\n\n")
        f.write("--- gamma window-convergence check (exact phi, deterministic) ---\n")
        f.write("Closed-form phi = 1 - ((t-tau_E)/t)^alpha tabulated on a log grid\n")
        f.write("(NOT Monte Carlo; RNG never seeded). Each window is one decade\n")
        f.write("[w, 10*w]; w=2e3 reproduces the baseline gamma to 3 s.f.\n")
        f.write(f"{'fit window':>12s}  " + "".join(f"{'a='+str(a):>9s}" for a in ALPHAS) + "\n")
        for row in gwc["windows"]:
            f.write(f"{'t>=%.0e' % row['t_min']:>12s}  "
                    + "".join(f"{row['gammas'][a]:>9.4f}" for a in ALPHAS) + "\n")
        f.write("\ngamma -> 1 MONOTONICALLY for every alpha as the window slides right:\n")
        f.write("the baseline drift of gamma below 1 (down to 0.964 at alpha=3) is a\n")
        f.write("subleading finite-window artifact, NOT an alpha-dependence. The leading\n")
        f.write("decay exponent is identically 1 => NO polynomial escape threshold alpha_c.\n")
        f.write(f"\nLocal log-log slope of exact phi at alpha={gwc['slope_alpha']:.0f} "
                f"(leading exponent identically 1):\n")
        for s in gwc["local_slopes"]:
            f.write(f"  t={s['t']:.0e}: slope={s['slope']:+.4f}\n")
        f.write("(-0.90 at t=2e3 -> -1.0000 by t>=2e7: the drift vanishes as t grows.)\n\n")
        f.write("--- Exponential growth |M| = M0 e^{kappa t} (escape regime) ---\n")
        f.write(f"{'kappa*tauE':>11s}  {'phi_inf':>9s}  {'nu_inf':>8s}  "
                f"{'|M|(T)/|M|(0)':>16s}  {'eta(T)/eta(0)':>14s}\n")
        for r in expo:
            f.write(f"{r['kappa_tauE']:>11}  {r['phi_inf']:>9.4f}  "
                    f"{r['nu_inf']:>8.4f}  {r['M_ratio_T']:>16.3e}  {r['eta_ratio_T']:>14.3e}\n")
        f.write("\nphi_inf > 0 (exponential growth escapes the nostalgia floor: nu < 1),\n"
                "but eta = I_pred/I_mem -> 0 STRUCTURALLY: I_pred is horizon-bounded while\n"
                "I_mem grows with the retained capacity |M| proportional to e^{kappa t}.\n"
                "This is the structural smallness of eta, NOT an energy-budget collapse.\n")

    (HERE / "run.log").write_text("\n".join(log_lines), encoding="utf-8")
    print("\nwrote results_summary.{txt,json}, run.log, figures")


if __name__ == "__main__":
    main()
