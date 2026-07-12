# `superlinear_memory` — phase boundary (Theorem 3, C3)

Article #3 (`landauer-undertow-oa`) § 5 / § 6 item 4 / supplementary § S4.4.

## What it computes

Memory capacity grows as `|M(t)| = M0·t^α` for `α ∈ {1, 1.5, 2, 3}`
(polynomial), and separately `|M(t)| = M0·e^{κt}` (exponential escape). FIFO
refresh. The controlling quantity is the **fresh-bit fraction**

```
φ(t) = (|M(t)| − |M(t−τ_E)|) / |M(t)| = 1 − ((t−τ_E)/t)^α  ≈  α·τ_E/t
```

We fit `φ(t) ~ t^{−γ}` on the tail `t ≥ 2000` and report:

- the decay exponent `γ` for each polynomial `α` (claim: `γ = 1` for **every**
  finite `α`, so the fresh fraction `φ(t) → 0`, hence **no polynomial `α_c`**).
  The tabulated `ν(T) = 1 − φ` is the *naive arithmetic per-bit* fraction: it
  → 1, but this is a **circular** threshold-model quantity (it bakes in
  bit-uniformity) and is **not** `ν^theor`. The load-bearing closure is
  `η_v = I_pred/I_mem → 0`; the true `ν^theor` is held at the C1 tracking floor
  (`ν^theor → ν_{C1} < 1`, not 1);
- the prefactor `φ·t → α·τ_E`;
- for exponential growth, the constant fresh-fraction limit `φ_∞ = 1 − e^{−κτ_E} > 0`
  (escape, `ν^theor < 1`) and the capacity ratio `|M(T)|/|M(0)|` — while a horizon-bounded
  numerator `I_pred ≤ H_τ` over a growing denominator `I_mem` drives `η_v = I_pred/I_mem → 0`
  structurally, by an *informational*, not energy-budget, route. (The true `I_mem ≤ H(X_E^{≤t})`
  grows only ~linearly; the run takes `I_mem ∝ |M|` as a conservative upper bound, so the reported
  η_v ratio = `|M(0)|/|M(T)|` over-states the decay — the true `η_v` falls ~1/t.)

## Parameters

- `k = 8`, `K = 56`, `τ_E = 200`, `M0 = 100`
- polynomial `α ∈ {1, 1.5, 2, 3}`; exponential `κτ_E ∈ {0.1, 0.5, 1.0}`
- `T = 20000`, fit window `t ≥ 2000`, 400 log-spaced measurement times
- `SEED = 20260529` (article #2 seed + 5) is **decorative / unused**: the output
  is a deterministic closed-form tabulation of `φ = 1 − ((t−τ_E)/t)^α` — there is
  **no Monte Carlo**, `np.random` is never called, the RNG is never seeded. The
  value is kept only for provenance.

## How to run

```
cd simulations/superlinear_memory
pip install -r requirements.txt
python main.py
```

Runtime < 1 s. Outputs: `results_summary.{txt,json}`, `run.log`,
`fig_superlinear_phi.png`, `fig_superlinear_escape.png`.

## Actual result (fixed seed)

Polynomial growth:

| α   | γ (φ~t^−γ) | expected | φ·t   | α·τ_E | ν(T)   |
|----:|-----------:|---------:|------:|------:|-------:|
| 1.0 | 1.0000     | 1.000    | 200.0 | 200   | 0.9900 |
| 1.5 | 0.9908     | 1.000    | 297.6 | 300   | 0.9850 |
| 2.0 | 0.9816     | 1.000    | 393.7 | 400   | 0.9801 |
| 3.0 | 0.9636     | 1.000    | 581.3 | 600   | 0.9703 |

Exponential escape:

| κτ_E | φ_∞    | ν_∞    | \|M(T)\|/\|M(0)\| |
|-----:|-------:|-------:|------------------:|
| 0.1  | 0.0952 | 0.9048 | 1.99e+04          |
| 0.5  | 0.3935 | 0.6065 | 3.14e+21          |
| 1.0  | 0.6321 | 0.3679 | 9.84e+42          |

- **γ = 1 independent of α: CONFIRMED.** γ ∈ {1.000, 0.991, 0.982, 0.964},
  spread 0.036, all ≈ 1. The decay exponent does **not** depend on α, so the
  fresh fraction `φ(t) → 0` for every finite polynomial α — **no polynomial
  escape threshold `α_c`**, matching Theorem 3. (The tabulated arithmetic
  `ν(T) = 1 − φ → 1` is the naive/circular per-bit majorant, **not** `ν^theor`:
  the paper's `ν^theor` is held at the C1 floor `→ ν_{C1} < 1`, not 1; the
  load-bearing closure of the route is `η_v → 0`.)
  The visible downward drift of γ to 0.964 at large α on the baseline window
  (`t ≥ 2000`) is a **finite-window subleading effect, not an α-dependence**: as
  the fit window slides right, γ → 1 for **all** α (see the *gamma
  window-convergence check* block in `results_summary.{txt,json}`).
- **Prefactor φ·t → α·τ_E: CONFIRMED** (200/298/394/581 vs 200/300/400/600).
- **Exponential escape: CONFIRMED.** `φ_∞ > 0` so `ν^theor` is held below 1, but
  the capacity `|M| ∝ e^{κt}` diverges (ratios up to 9.8e42 over the run) while the
  horizon-bounded `I_pred ≤ H_τ` is fixed, so `η_v = I_pred/I_mem → 0` structurally —
  escape from the nostalgia floor does not help, by an *informational* (not energy-budget)
  route (Theorem 3 § 5.4). NB the true `I_mem ≤ H(X_E^{≤t})` grows only ~linearly, so the
  true `η_v` decay is ~1/t; the capacity ratio is a conservative upper bound.

### Honest note — now verifiable in the simulation

The small downward drift of γ from 1.000 (α=1) to 0.964 (α=3) is a **finite-fit-
window artifact**: the exact `φ = 1 − ((t−τ_E)/t)^α` has sub-leading curvature
that flattens the log-log slope below 1 for larger α at finite t; the leading
asymptotic exponent is identically 1 for all α (the `φ·t → α·τ_E` prefactor check
confirms the `t^{−1}` leading order). It does not affect the qualitative
conclusion (γ → 1, no α_c).

This is no longer just asserted — `gamma_window_convergence()` **proves** it by
re-fitting γ on one-decade windows `[w, 10w]` of the exact φ with growing lower
bound `w` (`w=2000` reproduces the baseline fit):

| fit window | α=1.0 | α=1.5 | α=2.0 | α=3.0 |
|-----------:|------:|------:|------:|------:|
| t ≥ 2e3    | 1.000 | 0.991 | 0.982 | 0.963 |
| t ≥ 2e4    | 1.000 | 0.999 | 0.998 | 0.996 |
| t ≥ 2e5    | 1.000 | 1.000 | 1.000 | 1.000 |
| t ≥ 2e6    | 1.000 | 1.000 | 1.000 | 1.000 |

γ → 1 **monotonically for every α** as the window slides right. The local log-log
slope of the exact φ at α=3 runs `−0.897` (t=2e3) → `−0.990` → `−0.999` →
`−1.0000` (t ≥ 2e7): the leading-term exponent is identically 1, so the drift is
strictly subleading and there is **no α_c**.
