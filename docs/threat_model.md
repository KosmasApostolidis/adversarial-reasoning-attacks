# Threat Model

## Attacker capabilities

- **White-box access** to HuggingFace fp16 VLM weights (publicly available).
- Knowledge of the agent's system prompt and the JSON tool schema.
- Ability to perturb an input image at inference time (digital perturbation,
  L∞-bounded in pixel space).
- **No access** to the Ollama Q4 quantized server internals — treated as a
  black box for transfer evaluation.

## Attacker goals

1. **Untargeted** — maximise trajectory divergence from the benign-image
   agent behaviour, measured by tool-sequence Levenshtein distance.
2. **Targeted** — force selection of a specified incorrect tool at step *k*
   (e.g., `escalate_to_specialist` when the benign trajectory would have
   called `request_followup`).

## Perturbation constraints

- `‖δ‖∞ ≤ ε` with ε ∈ {2/255, 4/255, 8/255, 16/255}. `‖δ‖₂` variant reported
  as an ablation.
- Attacks are computed on the fp16 HF surrogate and delivered to the Ollama
  Q4 server as PNG-encoded base64 pixel arrays via the chat API. This is
  the same delivery path an adversary could realistically use.

## Out of scope

- Text-based jailbreaking or prompt injection (see Qi 2023, Zou 2023 for
  the adjacent literature).
- Attacks on the clinical workflow itself (sandboxed tool registry).
- Physical-world perturbations — digital only.
- Attacks on real patient data or real clinical decision support systems.

## Assumed safety bounds

- The sandboxed tool registry never triggers real-world side effects —
  `draft_report` returns a stub string, `request_followup` returns a dict.
- Attacks are evaluated only against the research benchmark and never
  deployed against a clinical system.
