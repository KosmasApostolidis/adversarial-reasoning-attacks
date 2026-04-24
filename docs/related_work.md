# Related Work

## Image-space attacks on VLMs

- **Qi et al., 2023** — *Visual Adversarial Examples Jailbreak Aligned LMs*.
  White-box image perturbations that induce unsafe text generation from
  aligned multimodal models. Our attack-side loss patterns (per-token CE
  on the generation tail) follow their formulation. Key difference: we
  target tool-call tokens, not free-text safety behaviour.
- **Zhao et al., 2024** — transfer attacks on multi-modal systems.

## Adversarial attacks in medical ML

- **Finlayson et al., 2019** — *Adversarial Attacks on Medical Machine
  Learning*. Seminal motivation for studying adversarial robustness in
  clinical ML; our benchmark extends the framing from single-task
  classifiers to tool-using agents.

## Attacks on tool-using agents

- **Fu et al., 2024** — *Imprompter: Tricking LLM Agents into Improper
  Tool Use*. Text adversarial suffixes that cause tool misuse.
- **InjecAgent** — evaluation benchmark for tool-use attacks via prompt
  injection.

## Gap this work fills

Prior work either (a) attacks VLM *text* output with images, or (b) attacks
tool-using *text* agents with text. No prior public benchmark measures
how white-box **image** perturbations alter the **tool-call trajectories**
of **agentic VLMs**, particularly with an explicit HF-fp16 → Ollama-Q4
transfer study in a medical imaging domain. Our contribution is the
measurement framework + baseline + transfer analysis in this specific
cell of the attack-surface matrix.

## Attack primitives reused

- **Custom PGD** — `src/adversarial_reasoning/attacks/pgd.py`. Self-contained
  L∞-PGD with teacher-forced cross-entropy against tool-call target tokens.
  (`torchattacks` was considered but its 3.5.x wheels pin
  `requests~=2.25.1`, conflicting with `datasets` / `smolagents`; dropped.)
- IBM Adversarial Robustness Toolbox — APGD / AutoAttack wrappers.
- Custom C&W (Phase 2): bespoke margin loss over tool-selection logits.

Our custom attacks (`trajectory_drift_pgd`, `targeted_tool_pgd`) extend
these primitives with bespoke losses over tool-call token sub-sequences.
