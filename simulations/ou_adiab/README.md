# `ou_adiab` — is `B → K/2` realisable under OU drift? (Theorem 1, C1)

Article #3 (`landauer-undertow-oa`) § 6.1 / supplementary § S4.1.

## What it tests

Theorem 1 derives, in the linear-Gaussian (Kalman) approximation, that the
cumulative excess loss of the ideal Bayesian learner tracking a slowly drifting
OU latent grows as

```
L_excess(t) = A·t + (K/2)·ln(λt) + C + o(1),   K = k(k-1) = 56,  K/2 = 28.
```

The open question (inherited from companion paper #2, § S8.3) was whether the
log-coefficient `B = K/2` can be **recovered numerically** as a slope: earlier
scans on a fixed window `[10³,10⁴]` left `B/(K/2) ≤ 0.43` and convergence open. A
natural hypothesis is that the scan never entered the tracking regime
`ε_track = I_rate·σ²/λ² ≪ 1`.

This experiment tests that hypothesis **in the exact linear-Gaussian setting the
theorem assumes** (so any failure is a property of the BNT log, not of a
suboptimal softmax learner). Model: `K` independent scalar OU coordinates,
Gaussian observations `y = θ + v` (`Var v = r`), and the **optimal Kalman
filter** — which *is* the ideal Bayesian learner. `r` fixes the per-step Fisher
information `I_rate = 1/r`, so `ε_track = σ²/(λ²r)`.

Three controlled experiments:

- **(A) Static anchor** (`λ=0, σ=0`, diffuse prior): the cumulative excess loss
  *must* be exactly `(K/2) ln(t)` (classical MDL/BIC redundancy). Validates the
  measurement pipeline and the `K/2` target.
- **(B) Local-slope diagnostic** under OU drift: track `dL_excess/d ln t` along
  the whole trajectory and locate where it equals `K/2`.
- **(C) Wide-window asymptotic scan** over `ε_track` (with `r = 10⁻²` so the
  window `[3τ_conc, 0.1/σ²]` has ratio ≈150, not ≈10): fit `A·t + B·ln(λt) + C`,
  report `B/(K/2)` with a bootstrap 95% CI and the regressor collinearity
  `corr(t, ln)`.

## Result (HONEST — convergence NOT achieved)

- **(A)** recovers `slope/(K/2) ≈ 0.99` (R² ≈ 0.9998): the BNT log is real and
  the pipeline is correct.
- **(B)** the local log-slope equals `K/2` **only during the brief initial
  prior-collapse burst** (`t` up to a few/`λ`). After that the drift caps the
  effective sample count at `n_eff ~ 1/g*`, `L_excess` **plateaus** at height
  `~ (K/2) ln(1/g*)`, and the slope collapses to ≈ 0.
- **(C)** in the wide asymptotic window `B/(K/2) ≈ 0` (the loss there is purely
  linear) and does **not** rise toward 1 as `ε_track → 0`. The "wrong regime,
  push `ε_track → 0`" hypothesis is **refuted**.

**Why convergence is structurally impossible here:** `τ_conc` (concentration
time `1/g* ≫ τ_E`, distinct from the forgetting scale `τ_f^forget ~ τ_E`)
and `1/σ²` (adiabatic validity bound) both scale as `1/σ²`, so the window where
the log lives (the transient burst, `t < τ_conc`) and the asymptotic requirement
`t ≫ τ_conc` cannot be separated into a wide log window. The analytic identity
`B = K/2` (the `ε_track → 0` limit) is realised only **transiently**; it is not
recoverable as a wide-window asymptotic slope under genuine OU drift, regardless
of how small `ε_track` is pushed. This is reported as a structural numerical
limitation of C1 in § 6.1 / § S4.1.

## Parameters

- `k = 8`, `K = 56`, `λ = 10⁻²`, diffuse prior `P0 = 10⁶`
- `r = 1` for (A)/(B); `r = 10⁻²` for the wide-window scan (C)
- `ε_track` scan: 10 log-spaced points over `[3·10⁻¹, 3·10⁻²]` — the range where
  the log-window stays wide and σ-invariant (`t_max = 0.1/σ² < 4·10⁶` cap)
- `N_runs = 40`, bootstrap `N = 2000`, seed `20260527`

## How to run

```
cd simulations/ou_adiab
pip install -r requirements.txt
python main.py
```

Runtime ≈ 7 min (dominated by the deepest scan point). Outputs:
`results_summary.{txt,json}`, `run.log`, `fig_ou_adiab_B_vs_eps.png`
(windowed `B/(K/2)` flat near 0 vs the static-anchor and burst references),
`fig_ou_adiab_Lexcess_scan.png` (local log-slope: `K/2` burst then plateau).
