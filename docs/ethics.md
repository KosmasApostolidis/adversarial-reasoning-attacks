# Ethics Statement

## Motivation

This benchmark is released to support **defensive** research: evaluating
and hardening multimodal medical AI agents against adversarial image
manipulation. The framing follows Finlayson et al. 2019 — understanding
the attack surface is a prerequisite for building robust clinical AI.

## Principles

1. **No clinical deployment.** The attacks here must not be used against a
   real clinical decision-support system, EHR, radiology PACS, or patient
   care workflow.
2. **Public / de-identified data only.** Primary benchmark uses ProstateX
   (public, TCIA). Workspace PCa MRI cases (de-identified, internal) are
   used only for ablation and are never released publicly.
3. **Sandboxed tools.** Every tool in the registry is a deterministic stub
   with no network calls and no real side effects. There is no path from a
   tool call in this benchmark to a real-world medical action.
4. **Disclosure before release.** If an attack is highly effective on a
   specific open-weights VLM, we will coordinate with the maintainers
   (Qwen, OpenBMB, Meta) ahead of publication.

## Misuse considerations

- **Could this lower the bar for an attacker?** The attack implementations
  themselves (PGD, APGD, C&W, custom trajectory-drift) are standard
  primitives that are already widely available. The novel contribution is
  the *measurement framework* (tool-call trajectory metrics + medical
  agent harness), which is more useful to defenders than to attackers.
- **Could this enable dataset poisoning?** No — the benchmark operates at
  inference time and does not produce training-time attacks.
- **Could this be reused for non-medical attacks?** Yes, but the relevant
  components (the agent harness, the metrics, the attack primitives) are
  all independently useful defensive tools, and the risk of reuse is low
  relative to the benefit of understanding the agentic-VLM attack surface.

## Disclaimer

**This code is not clinically validated software. Do not use it to inform
patient care decisions.**
