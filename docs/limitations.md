# Limitations

## 1. Baseline is VLM self-consistency, not medical ground truth

The "benign trajectory" against which we measure attack-induced divergence
is the VLM's own canonical behaviour on a clean image, not a clinically
validated correct action. This measures *reasoning instability under
perturbation*, not *medical correctness degradation*. We cannot claim
that an attacked trajectory is clinically wrong — only that it differs
from what the model would have done on the clean image.

## 2. Quantization gap

White-box attacks are computed on the HF fp16 surrogate and transferred
to Ollama Q4. Attack success retention on the Ollama backend is an
empirical quantity we measure (`attack_transfer_rate`); it is a lower
bound on the effect of the same attack against a gradient-accessible
attacker on the deployment backend.

## 3. Tool-calling maturity variance across VLMs

- Qwen2.5-VL-7B has native JSON function calling and is our anchor model.
- LLaVA-v1.6-Mistral-7B and Llama-3.2-Vision-11B use prompt-scaffolded ReAct-style
  tool use. The latter in particular has only text-official tool support
  from Meta; with vision input, tool use is not a guaranteed contract.
- Intra-seed trajectory variance at T=0 is measured per model (Phase 0
  noise floor) and used to gate what counts as a meaningful attack
  effect. Cross-model absolute effect sizes should be interpreted with
  per-model noise in mind.

## 4. Dataset scope

Primary benchmark is prostate MRI (ProstateX). Secondary is public
radiology VQA. Generalisation to chest X-ray, pathology, or ophthalmology
is not evaluated here.

## 5. Sandboxed tools are not real clinical actions

The six-tool registry is deterministic and offline. Real-world medical
agents would invoke heterogeneous APIs (PACS, EHR, guideline databases,
literature search) whose latency and failure modes would add noise this
benchmark does not capture.

## 6. Single-image scenarios

All current tasks operate on a single image (or a small multi-series
panel treated as one composite). Multi-study longitudinal workups are
out of scope.

## 7. Custom agent harness vs. `smolagents`

The plan locks `smolagents` as the agent framework. In practice we ship a
custom `MedicalAgent` (`src/adversarial_reasoning/agents/medical_agent.py`)
implementing the same ReAct-with-tool-registry loop. Reasons:

- Tighter control over the tool-call extraction surface (balanced-brace
  JSON scanner) for reproducible parsing across Qwen `<tool_call>` Hermes
  wrapping, LLaVA prompt-scaffolded JSON, and MLlama free-text emission.
- Fewer moving parts between the VLM forward pass and the trajectory that
  gets scored, which matters when measuring attack-induced *small* drifts.
- Zero dependency on smolagents' own LLM backends / chat-template glue.

We still list `smolagents` in the pinned deps so swapping the harness for
a cross-check ablation is a 1-file change if reviewers request it.

## 8. Attacker knowledge assumptions

We assume a white-box attacker. Black-box query-only attacks (e.g.,
transfer-only adversaries without any HF surrogate access) are not
studied. Transfer from HF fp16 to Ollama Q4 is reported but is itself a
partial black-box result, not a full query-only one.
