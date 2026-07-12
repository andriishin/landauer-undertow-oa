# `superlinear_noncirc` — non-circular C3 floor test (nostalgia plateaus at ν_C1 < 1)

Article #3 (`landauer-undertow-oa`), companion / repair to `superlinear_memory`
(supplementary §S4.4 / §S4.5). Definitions of ν, ν^theor, ν_C1, ε_track,
Σ_∞ follow [Andriishin2026b].

## What it tests

For a learner whose memory capacity grows **superlinearly**, `|M(t)| = M₀·tᵅ`
(`α ∈ {1, 1.5, 2, 3}`), does the *oracle-normalized nostalgia* ν^theor **plateau
at the C1 tracking floor** `ν_C1 < 1` — rather than drift to 1?

The existing `superlinear_memory` sim scores each retained snapshot with a
per-bit "useful iff age < τ_E" gate, so its tabulated per-bit majorant
`ν_naive = 1 − φ(t) → 1`. The paper itself flags this as **circular** (it bakes
in per-bit uniformity) and states that the true ν^theor is held at the C1 floor.
This sibling proves that directly: it computes ν^theor from a **real joint
Gaussian posterior**, with **no per-bit age gate**, and shows it plateaus at the
analytic `ν_C1 = Σ_∞/σ_0²`.

## Model (multiplicative — θ *is* the transition coupling, not an emission)

`K = 56` independent scalar coordinates. Per coordinate:

```
X_{t+1} = θ_t · X_t + w_t ,        w_t ~ N(0, q)        (state)
θ_{t+1} = (1−λ) · θ_t + u_t ,       u_t ~ N(0, σ²)       (OU drift of the coupling)
```

θ is observed **only multiplicatively**, through the state pair `(X_t, X_{t+1})`:
given `X_t`, the pair is a Gaussian observation of `θ_t` with design `X_t` and
noise `q`, i.e. per-step Fisher information `X_t²/q`. The Fisher rate
`I_rate = E[X_t²]/q` is **measured in-run** (≈ 1 for small θ, not assumed).
Parameters: `λ = 0.1`, `σ² = 0.002`, `q = 1` ⇒ adiabatic `σ_0² = σ²/(2λ) = 0.01`,
`ε_track = I_rate·σ²/λ² ≈ 0.20`. `|θ| ≪ 1` (adiabatic tails negligible), so the
linear-Gaussian **Kalman filter is the exact optimal learner**. `T = 10⁵`,
`R = 48` realizations, `M₀ = 100`, `τ_E = 1/λ = 10`, seed `20260527`, numpy only.

## The non-circular learner (the crux)

The growing FIFO retains the most recent `M(t) = M₀·tᵅ` pairs. From them the
learner forms a **real joint Gaussian posterior over the current θ_t** — a
recursive Kalman/RLS update in which each pair is a Gaussian likelihood in θ with
precision `X_s²/q`, and older retained pairs are down-weighted by the OU
forgetting (the predict step inflates the variance by `σ²` each step,
`τ_f ≈ τ_E = 1/λ`). Multiple **fresh** snapshots of the **same** coordinate
*jointly* concentrate the estimate toward `Σ_∞` — the drifted-out past is
uninformative about the current θ_t, so accumulating more (stale) memory does
**not** push Σ below `Σ_∞`. There is **no** "bit useful iff age < τ_E" gate.

Because `M₀·tᵅ ≥ t` for every `α ≥ 1` (with `M₀ ≥ 1`), the FIFO retains the
**entire history** for all α, so the joint posterior over θ_t is the full-memory
Kalman posterior for every α — hence **ν^theor is exactly α-independent**. What
sets the floor is the recent ~`τ_conc` window (see `fig_noncirc_window_floor.png`);
retaining more than that (as all these α do) does not lower Σ below `Σ_∞`.

## Analytic target (reproduced numerically, independent cross-check)

```
σ_0²      = σ²/(2λ)                                    (OU latent variance)
ε_track   = I_rate·σ²/λ²
Σ_∞       = (λ/I_rate)(√(1+ε_track) − 1)               (Riccati tracking floor)
ν_C1      = Σ_∞/σ_0² = (2/ε_track)(√(1+ε_track) − 1) ≈ 1 − ε_track/4  (≈ 0.954)
```

`ν_C1 < 1` is the **floor of nostalgia**: the optimal learner permanently
captures a residual fraction `1 − ν_C1 ≈ ε_track/4` of the oracle's predictive
information, forever — a nonzero floor, not total obsolescence.

## Three non-circular nostalgia measures (θ known by construction)

With `E[D_KL] = E[(θ_t − θ̂_t)²·X_t²]/(2q)` and the predicted estimate θ̂_t (the
learner must predict `X_{t+1}` before seeing it):

| measure | definition | role |
|---|---|---|
| (i) `ν_varratio` **PRIMARY** | `E[(θ−θ̂)²]/E[θ²]` | leading-order Fisher-canceling form = the paper's `Σ_∞/σ_0²` (§S1.3); low-variance, flat |
| (ii) `ν_predinfo` (literal) | `E[D_KL] / (½ E[−ln(1−θ²)])` | X²-weighted KL over per-pair mutual information (the task's explicit formula) |
| (iii) `ν_fullwtd` | `E[D_KL] / (E[θ²X²]/2q)` | fully X²-weighted; ~1.6% below ν_C1 because it retains a sub-leading design–error anticorrelation the closed form drops |

In the paper's §S1.3 KL expansion the Fisher `I` cancels (`½IΣ / ½Iσ_0² = Σ/σ_0²`),
so the **variance-ratio** measure is the faithful measured analog of `ν_C1`; the
two X²-weighted forms are refinements. All three **plateau** and **bracket ν_C1
to within ~1.6%**.

## How to run

```
cd simulations/superlinear_noncirc
pip install -r requirements.txt
MPLBACKEND=Agg python main.py
```

Runtime ≈ 7 s. Outputs: `results_summary.{txt,json}`, `run.log`,
`fig_noncirc_nu_plateau.png`, `fig_noncirc_two_track.png`,
`fig_noncirc_window_floor.png`.

## Result (fixed seed 20260527)

Measured (plateau over `t ≥ 10⁴`): `I_rate = 1.022`, `ε_track = 0.204`,
`ν_C1(cont) = 0.9535` (discrete brackets 0.9485 / 0.9583).

| ν^theor measure | plateau | vs ν_C1(cont) |
|---|---:|---:|
| (i) variance-ratio (primary) | **0.9628** | +0.97 % |
| (ii) predictive-info (literal) | 0.9700 | +1.73 % |
| (iii) fully X²-weighted | 0.9465 | −0.74 % |

- **(1) Plateau matches ν_C1: PASS.** The primary ν^theor is flat
  (slope `+1.2·10⁻³` per ln t; plateau halves 0.9619 / 0.9637) at
  **0.963 ≈ ν_C1 = 0.954** (<1 %), **α-independent** (FIFO = full history for all
  α). The three measures bracket ν_C1 within ~1.6 %.
- **(2) Σ̂ → Σ_∞ (floor, not 0): PASS.** Σ̂ = 0.01010, realized MSE = 0.01015 ≈
  the discrete Riccati root `Σ_∞ = 0.01009` — the learner reaches the **floor**,
  not zero.
- **(3) Sanity limits: PASS.** No-memory (θ̂ = 0) ⇒ ν = 1.000 (exact);
  single-snapshot ⇒ ν = 0.992 (~1) — reproduces the artifact. Oracle (θ̂ = θ) ⇒
  ν = 0.000 (exact).
- **(4) Naive majorant diverges: PASS.** `ν_naive = 1 − φ → 1` for every α
  (0.9997–0.9999 at T), rising **above** the 0.963 plateau — the headline
  divergence, confirming the old threshold `→ 1` is the circular artifact.
  Meanwhile `η_v = I_pred/I_mem → 0` (`I_mem ∝ M₀·tᵅ`) at an α-dependent rate.
- **(5) I_pred^opt > 0: PASS** (5.5·10⁻³, non-degenerate).

**Verdict.** Growing-memory nostalgia ν^theor **plateaus at the C1 floor
ν_C1 < 1** and is **α-independent**, matching the analytic `Σ_∞/σ_0²`; the
per-bit majorant `ν_naive → 1` is the circular artifact. `η_v = I_pred/I_mem → 0`
carries the actual route closure (two-track story). The learner reaches `Σ_∞`,
not 0.

### Honest notes

- The three nostalgia measures span ~1.6 % around ν_C1. This spread is the
  **sub-leading design–error correlation** (the adaptive Kalman filter tracks
  better exactly when `X_t²` is large, so X²-weighting shifts the captured
  fraction); the closed-form `ν_C1 = Σ_∞/σ_0²` drops it. The variance-ratio
  measure — the one the paper's derivation corresponds to — matches to <1 %.
- A small continuous-vs-discrete gap (`ν_C1` cont 0.9535 vs discrete-predicted
  0.9583) is the `O(λ)` time-discretization of the OU drift; the discrete
  Riccati root is reported alongside the continuous headline.
- Numerical step `τ = 1` (recommended) throughout; the filter is the exact
  linear-Gaussian optimum, so no estimator approximation enters ν^theor beyond
  Monte-Carlo noise (θ is known by construction, so `ν^theor = ν^op`).
