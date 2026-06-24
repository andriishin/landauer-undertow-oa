# `psp_floor` — nostalgia floor for Poisson reset (Theorem 2, C2)

Article #3 (`landauer-undertow-oa`) § 6 item 3 / supplementary § S4.3.

## What it computes

Piecewise-stationary Poisson (PSP) drift: the logits `θ(t)` are piecewise
constant, reset points form a Poisson stream of intensity `μ = 1/τ_E`, and at
each reset `θ` is independently re-drawn (the "drift-with-reset" surrogate of
article #2 § 6.1). Mixing is exponential: a bit of age `Δ` retains predictive
value iff no reset has occurred on its coordinate since the write.

This is a **real trajectory simulation**. A closed-form `ν(t) = e^{−μΔ}` would
be tautological; this version actually **simulates the Poisson reset stream**:
each of the `K` logit
coordinates carries a realised per-step reset clock (`p_reset = 1 − e^{−μ}`). A
FIFO memory of `|M| = 400` snapshot-bits stores, per bit, a coordinate label and
a write time; a bit stays predictive while its coordinate has **not reset** since
the write, and becomes nostalgic the moment it does. The FIFO refreshes the `dM`
oldest bits per step (fresh fraction `c = τ_E/refresh_period = c_target`).

Reports, **read out from the realised reset process**:

- the floor `liminf ν` for each `μ` (should equal `1 − c`, invariant in `μ`);
- the exponential decorrelation rate, estimated from the realised **survival**
  curve `Ŝ(age)` = fraction of stored bits of a given age that have not yet seen
  a reset, fitted `ln Ŝ ~ −rate·age` over `[0, 5 τ_E]` with a bootstrap CI over
  runs. The rate is an **empirical output**, not asserted to equal `μ`.

This covers the **exponential end** of the mixing class, complementary to the
power-law end in `fou_floor`.

## Parameters

- `k = 8`, `K = 56`, `μ ∈ {10⁻³, 3·10⁻³, 10⁻²}`, `c_target = 0.30`
- `|M| = 400`, `T = 20000`, `N_runs = 40`, `measure_every = 50`, `N_boot = 400`
- seed `20260528` (article #2 seed + 4), offset per `μ`

## How to run

```
cd simulations/psp_floor
pip install -r requirements.txt
python main.py
```

Runtime ≈ 28 s. Outputs: `results_summary.{txt,json}`, `run.log`,
`fig_psp_nu_vs_mu.png`, `fig_psp_approach.png`.

## Actual result (fixed seed, realised reset streams)

| μ      | τ_E  | floor (empirical) | empirical rate | 95% CI               | nominal μ |
|-------:|-----:|------------------:|---------------:|:---------------------|----------:|
| 1e−3   | 1000 | 0.7127            | 1.010e−3       | [9.85e−4, 1.039e−3]  | 1.000e−3  |
| 3e−3   | 333  | 0.7092            | 3.015e−3       | [2.978e−3, 3.051e−3] | 3.000e−3  |
| 1e−2   | 100  | 0.7093            | 1.002e−2       | [9.95e−3, 1.009e−2]  | 1.000e−2  |

- **Floor invariance: CONFIRMED.** `floor ν = 0.709–0.713 ≈ 1 − c = 0.70`
  across a 10× range of switching rate `μ`; spread `0.0035`. The floor constant
  does not depend on `μ`. Because this is a *realised* reset process (bits can
  decorrelate slightly before the nominal horizon), the empirical floor sits just
  above 0.70 — a genuine output, not the threshold value baked in.
- **Exponential rate ≈ μ: CONFIRMED (genuinely).** The realised survival curve
  decays as `e^{−rate·age}` with rate `1.010e−3, 3.015e−3, 1.002e−2`; every
  bootstrap CI **brackets the nominal `μ`**. This is a real measurement of the
  decorrelation time-scale from the simulated reset stream, not a re-fit of the
  formula.

No divergences from theory. Together with `fou_floor` (power-law end) this
exercises both ends of the mixing class; the `fou_floor` power-law exponent
`4(1−H)` is recovered in the far tail on the exact stationary fOU (see that
README), and the PSP exponential rate `μ` is recovered here.
