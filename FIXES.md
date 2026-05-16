# Phase-2 fixes — adversarial-reasoning-attacks

Companion to `REVIEW.md`. Documents what was changed, what evidence
exists that the change behaves as intended, and what was deliberately
deferred. Branch: `worktree-review-phase1`. Three commits ahead of
`origin/main`.

## Findings approved for fix (6 total)

| ID  | SEV      | Outcome                                           |
| --- | -------- | ------------------------------------------------- |
| C1  | CRITICAL | Fixed upstream by PR #24 (no work needed here)    |
| C2  | CRITICAL | Fixed upstream by PR #24 (no work needed here)    |
| C3  | HIGH     | Fixed upstream by PR #24 (no work needed here)    |
| C4  | HIGH     | Fixed in `090f083` (Path a — parameter removed)   |
| C5  | HIGH     | Fixed in `d91acde` (APGD step-0 pure sign-SGD)    |
| AML1| HIGH     | Fixed in `8dfa3b2` (new gradient-masking gate)    |

The Phase-1 review was written against local `main`, but
`EnterWorktree` branched from `origin/main`, which already contained
PR #24's fixes. The post-discovery audit confirmed: C1's ε-domain
clarification (now uses pixel-domain ε with internal rescale via
``vlm.pixel_std``), C2's KL direction fix (manual
``KL(p_attacked || p_benign)``), and C3's ε=0 short-circuit
(``loss_final=0.0, success=True``) all land in `origin/main` exactly as
the user-selected design choices specified. C4, C5, and AML1 were
genuinely unfixed and were addressed in this branch.

---

## C4 — Remove `TargetedToolPGD.target_step_k` (commit `090f083`)

### What changed
- Dropped the ``target_step_k: int = 0`` dataclass field from
  `src/adversarial_reasoning/attacks/targeted_tool.py`.
- Dropped the matching metadata stamp
  ``result.metadata["target_step_k"] = ...``.
- Removed the parameter from
  ``runner.attacks.build_attack``,
  ``runner.attacks._dispatch_attack``,
  ``runner.attacks._reshape_and_reinfer``,
  ``runner.attacks.run_gradient_attack``.
- Removed the ``--target-step-k`` CLI flag.
- Stripped ``target_step_k:`` from 4 YAML configs
  (`configs/attacks.yaml`, `targeted_tool_smoke.yaml`,
  `targeted_tool_smoke_paired.yaml`, `targeted_tool_sweep.yaml`).
- Updated docs: ``docs/PROJECT_REPORT.md``, ``STUDY_OVERVIEW.md``.

### Why
The field was metadata-only — never threaded into the target-token
builder or loss construction. The attack always forces the target tool
as the *next* tool call (position 0 of the appended target sequence)
regardless of the configured step index. Documenting "step index"
while silently ignoring it is a paper-honesty problem.

### Tests
- Updated `tests/test_runner_attacks.py` to drop the parameter from
  ``build_attack`` call and the ``target_step_k`` metadata assertion.
- Updated `tests/test_coverage_gaps.py` likewise.
- Full pytest run: **378/378 pass** post-fix.

### Deferred
A paper-honest "step-k positional targeting" implementation (roll out
``k`` benign tool calls under no-grad, capture context tokens,
teacher-force the target at ``t_prompt + cumulative_context_len``) was
explicitly out-of-scope for this surgical removal. That's the Path-(b)
follow-up if the threat model ever genuinely needs step-k targeting.

---

## C5 — APGD step-0 pure sign-SGD per Croce-Hein 2020 §3.2 (commit `d91acde`)

### What changed
- `src/adversarial_reasoning/attacks/apgd.py`:
  threaded a ``step: int`` parameter into ``_apply_momentum_update``;
  branched the update rule so step ``k=0`` is pure sign-SGD
  (``x_new = z``) and the heavy-ball recurrence is applied only at
  ``k ≥ 1``.

### Why
Algorithm 1 line 6 in Croce & Hein 2020 specifies
``x^(1) = P(x^(0) + η·sign(∇f(x^(0))))`` — pure sign-SGD on the first
iterate. The heavy-ball recurrence (line 8) only kicks in at ``k ≥ 1``.
The previous code applied the momentum mix unconditionally; combined
with ``x_prev == x_curr`` at initialisation, that muted the step-0
update by a factor of ``momentum`` (=0.75 by default), so APGD
effectively took a 0.75·η first step instead of the prescribed η. This
systematically underutilised the ε-budget on the first iterate and
biased the checkpoint stagnation detector against APGD.

### Tests
Added `class TestApgdStepZeroPureSignSgd` to `tests/test_apgd.py`
covering two invariants on `_apply_momentum_update` directly:

1. **`test_step_zero_is_pure_sign_sgd_magnitude`** — with grad=ones,
   x_prev = x_curr, step=0, η=0.1, asserts
   ``delta = -η · ones`` (full step, no momentum mute).
2. **`test_step_one_applies_momentum_when_x_prev_equals_current`** —
   identical setup but step=1, asserts
   ``delta = -momentum·η·ones`` (recurrence engaged).

Full ``test_apgd.py`` run: **22/22 pass** post-fix.

### Deferred
An end-to-end "1-step APGD saturates ε-ball" assertion was
dropped — it conflated the ε-ball projection with the [0,1] image
clamp and was flaky against specific gradient/init alignments. The two
unit tests pin the exact step-0 invariant directly on the method.

---

## AML1 — Athalye-Carlini-Wagner 2018 gradient-masking gate (commit `8dfa3b2`)

### What changed
- New module `src/adversarial_reasoning/gates/gradient_masking.py`
  with ``GradientMaskingResult`` + ``run_gradient_masking`` +
  ``write_gate_report``.
- Wired into `src/adversarial_reasoning/gates/__init__.py`'s PEP-562
  lazy-import dispatcher.

### Why
A gradient-based attack reports honest robustness numbers only if the
loss surface actually exposes useful gradients. Athalye, Carlini &
Wagner 2018 ("Obfuscated gradients give a false sense of security")
list four canonical sanity checks; without these the benchmark can
silently mistake gradient masking for robustness. The prior Phase-0
gate set covered preprocessing transfer and intra-seed noise floor
but had no obfuscation check.

The gate runs four pass/fail invariants:

| Check                       | Pass criterion                                       |
| --------------------------- | ---------------------------------------------------- |
| (a) huge-ε loss drop        | ``Δloss ≥ 0.5 · \|benign\|`` at ε → "huge"           |
| (b) PGD beats noise         | PGD final loss ≤ uniform-L∞-noise loss at same ε     |
| (c) Loss monotonicity       | ≥ 80% of consecutive iterates non-increasing         |
| (d) Grad norm not collapsed | ``\|\|∇L\|\|_attacked ≥ 0.1 · \|\|∇L\|\|_benign``        |

The gate consumes pre-collected telemetry rather than driving its own
PGD loop — matches how `noise_floor` decouples agent execution from
the verdict and keeps the gate testable without a real VLM.

### Tests
Added 7 tests to `tests/test_gates.py`:
- Healthy run passes all four checks.
- Each of the four failure modes individually flips ``passes`` → False.
- JSON serialisation round-trips the verdict.
- `_is_monotonic` threshold boundary (exactly 80% non-increasing
  passes; below fails; empty/length-1 trajectories fail).

Full pytest run: **385/385 pass** post-fix.

### Deferred
- **Caller wiring**: nothing in the runner currently calls
  `run_gradient_masking`. Plumbing the gate into the per-cell PGD/APGD
  runs (logging loss trajectory + gradient norms) was out of scope for
  the surgical "add the gate" finding. Follow-up is a one-liner per
  attack runner once a benchmark sweep starts emitting telemetry.
- **Black-box vs white-box transfer check**: Athalye's fifth canonical
  check (black-box ASR ≤ white-box ASR) is not covered here because
  black-box is the Ollama-backed transfer eval, which sits in a
  different module. Adding it would require cross-module telemetry
  plumbing.

---

## Phase-1 findings deferred (not approved this round)

All MEDIUM and LOW findings from `REVIEW.md` (9 + 9 = 18) are
unchanged. Two of them (AR2, A1) overlap with C4 and would be
auto-resolved by the C4 commit; the others stand. Re-survey of those
findings against `origin/main` is a separate review pass.

## Test summary

| Test file               | Before       | After   |
| ----------------------- | ------------ | ------- |
| `tests/` (full suite)   | 378 passing  | 385 passing |
| C5 regression           | 0 tests      | 2 new tests, both green |
| AML1 gate               | 0 tests      | 7 new tests, all green |
| C4-affected tests       | 2 stale assertions | 2 updated, green |

No previously-passing test was disabled, skipped, or weakened.

## Commit list (3)

```
d91acde fix(apgd): apply pure sign-SGD at step 0 per Croce-Hein 2020 §3.2
090f083 refactor(targeted_tool): drop unused target_step_k field and CLI arg
8dfa3b2 feat(gates): add Athalye-Carlini-Wagner 2018 gradient-masking gate
```

Each commit message contains the per-finding rationale and is the
authoritative record. This file is a roll-up index.
