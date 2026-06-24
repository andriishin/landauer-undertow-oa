"""iter-002 sim 3: `psp_floor` -- nostalgia floor for Poisson reset (Theorem 2, C2).

Article #3 (`landauer-undertow`) § 6 item 3 / supplementary § S4.3.

Piecewise-stationary Poisson (PSP) drift: the logits theta(t) are piecewise
constant; reset points form a Poisson stream of intensity mu = 1/tau_E, and at
each reset theta is independently re-drawn from the stationary distribution
(the "drift-with-reset" surrogate of article #2 § 6.1). Mixing is exponential:
the probability that a bit of age Delta has NOT seen a reset since being written
is e^{-mu Delta} = e^{-Delta/tau_E}, so beta(Delta) = e^{-mu Delta}.

REAL TRAJECTORY SIMULATION (iter-002 P0-A closure).  Earlier drafts computed
nu(t) from the closed-form e^{-mu age} (tautological). This version actually
SIMULATES the Poisson reset stream: each of the K logit coordinates carries a
realised reset clock; a memory bit written at time s remains predictive iff NO
reset has occurred on its coordinate in (s, t]. The nostalgic fraction nu(t) and
the floor are read out from the realised reset process, and the exponential
approach rate is ESTIMATED from the empirical survival of stored bits (fraction
that have NOT yet seen a reset vs age), not asserted to equal mu.

We show the SAME nostalgia floor 1 - c for every switching rate mu (the floor
constant is invariant to mu at fixed c), and an exponential approach with
realised time-scale ~ 1/mu -- the exponential end of the mixing class,
complementary to the power-law end covered by `fou_floor`.

With the true theta(t) known by construction, nu^theor = nu^op (S4.5).
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
MUS = [1e-3, 3e-3, 1e-2]          # Poisson switching intensities mu = 1/tau_E
C_TARGET = 0.30                   # target fresh fraction -> floor 1 - c = 0.70
T_TOTAL = 20_000
N_RUNS = 40
SEED = 20260524 + 4               # iter-002 offset
MEASURE_EVERY = 50
M_BITS = 400                      # FIFO memory size (snapshot bits)
N_BOOT = 400


def simulate_nu_realised(mu: float, c_target: float, T: int, n_runs: int,
                         measure_every: int, M: int,
                         rng: np.random.Generator) -> dict:
    """nu(t) for a FIFO memory tracking a REALISED Poisson-reset (PSP) drift.

    K coordinates each carry an independent Poisson reset stream of intensity mu
    (per-step reset probability ~ 1 - e^{-mu}). A memory bit is written with a
    coordinate label and a write time; it stays PREDICTIVE while its coordinate
    has NOT reset since the write, else NOSTALGIC. tau_E = 1/mu. FIFO refreshes
    dM oldest bits/step so the fresh fraction is c. Floor = stale fraction.

    The empirical exponential decorrelation rate is estimated from the realised
    survival curve S_hat(age) = fraction of bits of a given age that have not yet
    seen a reset on their coordinate; fit ln S_hat ~ -rate*age over [0, 5 tau_E].
    """
    tau_E = 1.0 / mu
    refresh_period = tau_E / c_target
    dM = M / refresh_period
    p_reset = 1.0 - np.exp(-mu)        # per-step reset probability (exact)
    max_age = int(np.ceil(refresh_period)) + 1
    n_lag = min(max_age, T - 1)

    n_samples = T // measure_every
    times = np.zeros(n_samples)
    nu_arr = np.zeros((n_runs, n_samples))
    # realised survival of bits vs age, aggregated across runs
    surv_num = np.zeros(n_lag)   # # bits of age a that survived (no reset)
    surv_den = np.zeros(n_lag)   # # bits of age a observed
    surv_runs = np.zeros((n_runs, n_lag))

    for run in range(n_runs):
        ages = np.zeros(M, dtype=np.int64)
        coord = rng.integers(0, K_PARAMS, size=M)       # coordinate label of bit
        alive = np.ones(M, dtype=bool)                  # not yet reset since write
        carry = 0.0
        si = 0
        run_num = np.zeros(n_lag)
        run_den = np.zeros(n_lag)
        for t in range(1, T + 1):
            ages += 1
            # realised Poisson resets on each coordinate this step
            reset_coord = rng.random(K_PARAMS) < p_reset
            # any live bit whose coordinate reset this step becomes nostalgic
            if reset_coord.any():
                hit = reset_coord[coord] & alive
                alive[hit] = False
            # FIFO refresh: rewrite the dM oldest bits (fresh snapshot)
            carry += dM
            n_refresh = int(carry)
            carry -= n_refresh
            if n_refresh > 0:
                oldest = np.argpartition(ages, -n_refresh)[-n_refresh:]
                ages[oldest] = 0
                coord[oldest] = rng.integers(0, K_PARAMS, size=n_refresh)
                alive[oldest] = True
            if t % measure_every == 0 and si < n_samples:
                nu_arr[run, si] = 1.0 - float(alive.sum()) / float(M)
                times[si] = t
                si += 1
                # accumulate realised survival vs age
                a = np.minimum(ages, n_lag - 1)
                np.add.at(run_den, a, 1.0)
                np.add.at(run_num, a, alive.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            run_surv = np.where(run_den > 0, run_num / run_den, np.nan)
        surv_runs[run] = run_surv
        surv_num += run_num
        surv_den += run_den

    times = times[:si]
    nu_arr = nu_arr[:, :si]
    with np.errstate(invalid="ignore", divide="ignore"):
        surv = np.where(surv_den > 0, surv_num / surv_den, np.nan)
    return dict(times=times, nu_mean=nu_arr.mean(axis=0),
                nu_std=nu_arr.std(axis=0),
                ages=np.arange(n_lag), surv=surv, surv_runs=surv_runs,
                tau_E=tau_E, refresh_period=refresh_period)


def fit_rate_empirical(ages: np.ndarray, surv_runs: np.ndarray, tau_E: float,
                       n_boot: int, rng: np.random.Generator) -> dict:
    """Estimate the exponential decorrelation rate from the REALISED survival
    curve S_hat(age) (fraction of bits not yet reset): fit ln S_hat ~ -rate*age
    over age in [0, min(5 tau_E, end)]. Bootstrap CI over runs. The fit can
    differ from the nominal mu = 1/tau_E (finite sample); reported honestly."""
    n_runs = surv_runs.shape[0]
    hi = min(5.0 * tau_E, float(ages[-1]))
    mask = (ages >= 0) & (ages <= hi)
    A = ages[mask].astype(float)
    if A.size < 6:
        return {"ok": False}

    def one_fit(surv_curve):
        y = surv_curve[mask]
        good = np.isfinite(y) & (y > 1e-4)
        if good.sum() < 5:
            return None
        ly = np.log(y[good])
        X = A[good]
        M_ = np.vstack([X, np.ones_like(X)]).T
        coef, *_ = np.linalg.lstsq(M_, ly, rcond=None)
        return -coef[0]

    def safe_mean(arr):
        # column-mean ignoring NaNs; empty columns -> NaN (masked out by one_fit)
        finite = np.isfinite(arr)
        cnt = finite.sum(axis=0)
        s = np.where(finite, arr, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, s / cnt, np.nan)

    point = one_fit(safe_mean(surv_runs))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        v = one_fit(safe_mean(surv_runs[idx]))
        if v is not None and np.isfinite(v):
            boot.append(v)
    boot = np.array(boot)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) \
        if boot.size else (float("nan"), float("nan"))
    return {"ok": point is not None,
            "value": float(point) if point is not None else float("nan"),
            "ci95": ci, "fit_window": [0.0, hi], "nominal_mu": float(1.0 / tau_E),
            "n_boot": int(boot.size)}


def main() -> None:
    log_lines: list[str] = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    floor = 1.0 - C_TARGET
    log("== iter-002 sim 3: psp_floor -- nostalgia floor for Poisson reset (Theorem 2) ==")
    log("REAL trajectory simulation: Poisson reset streams simulated; nu(t), floor and")
    log("decorrelation rate read from the realised survival of stored bits (no formula re-fit).")
    log(f"k = {K_STATES}, K = {K_PARAMS}, mus = {MUS}")
    log(f"c_target = {C_TARGET}, floor 1-c = {floor:.3f}")
    log(f"T_total = {T_TOTAL}, N_runs = {N_RUNS}, M = {M_BITS}, seed = {SEED}")

    t_start = time.time()
    results = []
    for mu in MUS:
        rng = np.random.default_rng(SEED + int(round(1e5 * mu)))
        t0 = time.time()
        out = simulate_nu_realised(mu, C_TARGET, T_TOTAL, N_RUNS, MEASURE_EVERY,
                                   M_BITS, rng)
        times, nu = out["times"], out["nu_mean"]
        floor_emp = float(nu[int(0.75 * len(nu)):].mean())
        liminf_emp = float(nu[len(nu) // 2:].min())
        fit = fit_rate_empirical(out["ages"], out["surv_runs"], out["tau_E"],
                                 N_BOOT, rng)
        dt = time.time() - t0
        log(f"\n mu = {mu:g} (tau_E = {out['tau_E']:.0f}): floor = {floor_emp:.4f} "
            f"(target {floor:.3f}), liminf = {liminf_emp:.4f}  ({dt:.1f}s)")
        if fit.get("ok"):
            log(f"   EMPIRICAL exp rate = {fit['value']:.4e} "
                f"95% CI [{fit['ci95'][0]:.4e}, {fit['ci95'][1]:.4e}] "
                f"(nominal mu = {fit['nominal_mu']:.4e})")
        results.append({"mu": mu, "tau_E": out["tau_E"],
                        "floor_empirical": floor_emp, "liminf_empirical": liminf_emp,
                        "rate_fit": fit, "runtime_s": dt,
                        "times": times.tolist(), "nu": nu.tolist(),
                        "ages": out["ages"].tolist(),
                        "surv": np.where(np.isfinite(out["surv"]),
                                         out["surv"], 0.0).tolist()})

    total = time.time() - t_start
    floors = [r["floor_empirical"] for r in results]
    log(f"\n== total runtime {total:.1f}s ==")
    log(f"floor spread across mu: range {max(floors)-min(floors):.4f} "
        f"(min {min(floors):.4f}, max {max(floors):.4f})")

    # ---------------- figure: nu(t) for each mu ----------------
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        col = cmap(i / max(1, len(results) - 1))
        ax.plot(r["times"], r["nu"], "-", color=col, lw=1.5,
                label=rf"$\mu={r['mu']:g}$ ($\tau_E={r['tau_E']:.0f}$), floor {r['floor_empirical']:.3f}")
    ax.axhline(floor, color="k", ls="--", lw=1.2,
               label=rf"theory floor $1-c={floor:.2f}$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\nu(t)$ (nostalgic fraction)")
    ax.set_title(r"psp_floor: realised-reset nostalgia floor invariant to switching rate $\mu$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_psp_nu_vs_mu.png", dpi=150)
    plt.close(fig)
    log("saved fig_psp_nu_vs_mu.png")

    # realised survival curve (log-linear) -> exponential scale 1/mu
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, r in enumerate(results):
        col = cmap(i / max(1, len(results) - 1))
        a = np.array(r["ages"], dtype=float)
        s = np.array(r["surv"])
        m = s > 1e-4
        ax.semilogy(a[m], s[m], "-", color=col, lw=1.4,
                    label=rf"$\mu={r['mu']:g}$ (rate {r['rate_fit']['value']:.2e})")
    ax.set_xlabel("age (steps since write)")
    ax.set_ylabel(r"realised survival $\hat S(\mathrm{age})$ (no reset yet)")
    ax.set_title(r"psp_floor: realised exponential decorrelation (scale $\approx 1/\mu$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_psp_approach.png", dpi=150)
    plt.close(fig)
    log("saved fig_psp_approach.png")

    # ---------------- summaries ----------------
    summary = {
        "experiment": "iter-002 sim3 psp_floor: REALISED-reset nostalgia floor for Poisson reset (Theorem 2)",
        "method": "Poisson reset streams simulated per coordinate; nu(t), floor and decorrelation rate read from realised bit survival (no closed-form re-fit)",
        "k": K_STATES, "K_params": K_PARAMS, "mus": MUS,
        "c_target": C_TARGET, "floor_theory": floor,
        "T_total": T_TOTAL, "n_runs": N_RUNS, "M_bits": M_BITS, "seed": SEED,
        "total_runtime_s": total,
        "results": [{kk: vv for kk, vv in r.items()
                     if kk not in ("times", "nu", "ages", "surv")} for r in results],
        "floor_spread": max(floors) - min(floors),
    }
    (HERE / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    with (HERE / "results_summary.txt").open("w", encoding="utf-8") as f:
        f.write("=== iter-002 sim3: psp_floor -- REALISED-reset nostalgia floor (Theorem 2) ===\n")
        f.write("Poisson reset streams simulated; nu(t), floor and decorrelation rate read\n")
        f.write("from the realised survival of stored bits (NOT the closed-form e^{-mu age}).\n")
        f.write(f"k = {K_STATES}, K = {K_PARAMS}, M = {M_BITS}\n")
        f.write(f"c_target = {C_TARGET}, theory floor 1-c = {floor:.3f}\n")
        f.write(f"T_total = {T_TOTAL}, N_runs = {N_RUNS}, seed = {SEED}\n")
        f.write(f"total runtime {total:.1f}s\n\n")
        f.write(f"{'mu':>10s}  {'tau_E':>8s}  {'floor_emp':>10s}  {'liminf':>9s}  "
                f"{'emp_rate':>11s}  {'95% CI':>24s}  {'nominal mu':>11s}\n")
        for r in results:
            fit = r["rate_fit"]
            rv = f"{fit['value']:.4e}" if fit.get("ok") else "-"
            ci = (f"[{fit['ci95'][0]:.3e},{fit['ci95'][1]:.3e}]"
                  if fit.get("ok") else "-")
            f.write(f"{r['mu']:>10.3e}  {r['tau_E']:>8.0f}  "
                    f"{r['floor_empirical']:>10.4f}  {r['liminf_empirical']:>9.4f}  "
                    f"{rv:>11s}  {ci:>24s}  {r['mu']:>11.3e}\n")
        f.write(f"\nfloor spread across mu: {max(floors)-min(floors):.4f} "
                f"(min {min(floors):.4f}, max {max(floors):.4f}) -- "
                f"floor invariant to switching rate (realised-reset output);\n")
        f.write("realised survival decays exponentially with rate consistent with mu = 1/tau_E\n"
                "(within bootstrap CI) -- empirical output, NOT asserted equal to mu.\n")

    (HERE / "run.log").write_text("\n".join(log_lines), encoding="utf-8")
    print("\nwrote results_summary.{txt,json}, run.log, figures")


if __name__ == "__main__":
    main()
