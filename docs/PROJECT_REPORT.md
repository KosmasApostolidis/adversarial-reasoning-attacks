# Project Report — Adversarial Reasoning Attacks on Medical VLM Agents

**Repo:** `adversarial-reasoning-attacks`
**Date:** 2026-04-25
**Branch:** `main` @ `205d6b2`

---

## 1. Research question

Modern medical-imaging assistants are wired as **tool-using VLM agents**: a vision-language model receives an image plus a prompt, then drives a multi-step ReAct loop that invokes domain tools (`query_guidelines`, `calculate_risk_score`, `draft_report`, ...). The model is no longer producing a single answer — it is producing a *trajectory* of tool invocations.

We ask: **does an ε-bounded perturbation of the input image alone systematically warp the agent's tool-call trajectory?** If yes, the attack surface for adversaries is much larger than next-token classification: it includes *which sub-routine the agent calls and in what order*, which has direct clinical consequences (skipping `calculate_risk_score`, hallucinating `escalate_to_specialist`, etc.).

Phase 0 (sanity gates) + Phase 1 (full attack suite, ε × seed sweeps) are complete. Phase 2 (defenses, transfer to closed APIs) is future work.

Existing work on adversarial perturbations of VLMs targets the *next-token output* (forced-string image hijacks, jailbreak compliance). Existing work on attacking tool-using LLM agents operates through the *text channel* (prompt injection, memory poisoning, tool-description tampering). Existing work on medical-VLM robustness measures *classification or retrieval accuracy* on single-shot inference. No prior thread targets the **tool-call trajectory of a medical VLM agent under pixel-bounded gradient perturbation** — the threat model when a clinician uploads an image to a tool-using assistant. This work lives at that intersection.

---

## 1.5 Related work and threat-model positioning

The literature relevant to this report splits into three threads. We summarize each, then state precisely what each thread does *not* cover, which is where this work sits.

### 1.5.1 Adversarial attacks on VLMs (image channel)

| Work | What it does | What it does *not* do |
|---|---|---|
| Bagdasaryan et al. 2023 (*Abusing Images and Sounds for Indirect Instruction Injection*); Schlarmann & Hein 2023 (*On the Adversarial Robustness of Multi-Modal Foundation Models*, ICCV-W) | force VLM to emit attacker-chosen string from a captioning prompt | trajectory-level multi-step output |
| Carlini et al. 2023 (*Are aligned neural networks adversarially aligned?*); Qi et al. 2023 (*Visual Adversarial Examples Jailbreak Aligned LLMs*, AAAI 2024) | visual jailbreaks of aligned VLMs | tool-using agent setting |
| Bailey et al. 2024 (*Image Hijacks: Adversarial Images can Control Generative Models at Runtime*, ICML) | runtime control via specific-string / jailbreak / leak-context image hijacks | multi-tool reasoning loop |
| AdvDiffVLM, AdvCLIP, AdvEncoder, GAP | universal / transferable image perturbations against VLM encoders | task-specific tool-trajectory targeting |
| Schlarmann et al. 2024 (*RobustVLM*, ICML) | unsupervised adversarial fine-tuning of vision encoders (defense) | (defense-side, complementary to our work) |
| Liu et al. 2024 (ACM Computing Surveys 2024); *Awesome-LVLM-Attack* | survey + curated reading list | — |

**Limitation w.r.t. our work.** All of the above target the *single-utterance output* of the VLM (a string the attacker dictates, or refusal-bypass behaviour), not the multi-step tool-call sequence emitted by an agent loop. Their gradient signal is over the next token; ours is over a *trajectory* of tool invocations spanning up to 8 ReAct steps.

### 1.5.2 Adversarial attacks on tool-using LLM agents (text channel)

| Work | Channel | Target |
|---|---|---|
| Greshake et al. 2023 (*Not what you've signed up for*) | indirect prompt injection | text |
| AgentDojo (Debenedetti et al. 2024); InjecAgent (Zhan et al. 2024); BadAgent (Wang et al. 2024) | prompt / memory / tool-description injection | text |
| ASB — Agent Security Bench (Zhang et al., ICLR 2025) | DPI / IPI / memory-poisoning / Plan-of-Thought backdoor | text |
| AgentHarm (Andriushchenko et al., ICLR 2025) | harmful-task benchmark for tool-using agents | text |
| Foot-in-the-Door on ReAct (Oct 2024); *From Allies to Adversaries* (Dec 2024) | adversarial tool-call injection through prompt or tool registry | text |
| Wu et al. 2025 (*Dissecting Adversarial Robustness of Multimodal LM Agents*, ICLR 2025; `agent-attack`) | multimodal LM agents on general web-navigation tasks | image+text, web agents |

**Limitation w.r.t. our work.** These attacks inject through the **text** channel — the attacker controls some prompt, memory entry, or tool description. Our attack is **pixel-only**, ε-bounded in raw image space, and assumes no text-channel access. Wu et al. 2025 (`agent-attack`) is the closest neighbour but targets general web-navigation agents (e.g. VisualWebArena), not clinical tool-using agents, and uses task-success rate rather than trajectory edit-distance as the metric. Their threat model also permits modifying both image and text inputs; ours fixes the prompt.

### 1.5.3 Adversarial robustness in medical VLMs

| Work | Setup | Metric |
|---|---|---|
| Finlayson et al. 2019 (*Adversarial attacks on medical machine learning*, Science) | classification CNNs on dermoscopy / fundus / chest X-ray | clean→adversarial accuracy drop |
| CARES (May 2025, arXiv 2505.11413) | clinical safety benchmark — 18k prompts across 8 safety principles | harmful content / jailbreak / false-refusal rate |
| CoDA (chain-of-distribution) | clinical pipeline-shift framework with plausible perturbations | task accuracy under shift |
| PromptSmooth++ (MICCAI 2025) | randomized-smoothing certified robustness for frozen Med-VLMs | certified L2 radius |
| MFHA (Sci. Rep. 2025) | multimodal feature-heterogeneity attack | transferability + diagnostic-tool framing |
| Federated Med-VLM vulnerabilities (Nat. Sci. Rep. 2026) | federated-training threat model | classification accuracy |
| *On the Robustness of Medical VLMs* (Springer 2025) | generalization under domain shift across MIMIC / CheXpert / NIH | retrieval / classification accuracy |
| *Robustness in deep learning models for medical diagnostics* (AI Review, Springer 2024) | survey | — |

**Limitation w.r.t. our work.** These measure single-shot classification / retrieval / corruption robustness on a fixed task. None of them treat the model as an agent that emits a *tool-call trajectory*, and none measure trajectory edit-distance.

### 1.5.4 The gap this work targets

Our combination is, to our knowledge, novel. The contribution lives at the intersection of three threads, which is unoccupied:

| Dimension | This work |
|---|---|
| Input channel | ε-bounded gradient perturbation of pixel input only (no text-channel access) |
| Target system | Tool-using medical VLM agent (ReAct loop, deterministic clinical tools) |
| Attack objective | Trajectory disruption — *which* tool is invoked and *in what order* |
| Primary metric | Normalized tool-name edit distance; targeted-hit rate as secondary |
| Scope | Cross-model parity (Qwen2.5-VL-7B vs LLaVA-v1.6-Mistral-7B) under identical ε convention |
| Threat model | "Attacker controls the uploaded image, not the system prompt or the tool registry" — realistic for a clinical upload portal |

Restating: prior image-channel VLM attacks (§1.5.1) optimise next-token output but ignore the agent loop; prior agent attacks (§1.5.2) require text-channel access and ignore pixel-space gradients; prior Med-VLM robustness work (§1.5.3) measures classification accuracy and ignores tool trajectories. No prior thread covers all three. This work treats the tool-call sequence itself as the adversary's optimisation target, under the most restrictive (pixel-only) input channel, on a clinically-shaped task surface.

---

## 2. System under test

### 2.1 Models (`src/adversarial_reasoning/models/`)

| Loader | HF ID | Role |
|---|---|---|
| `QwenVL` | `Qwen/Qwen2.5-VL-7B-Instruct` | primary VLM, native function-calling |
| `LlavaNext` | `llava-hf/llava-v1.6-mistral-7b-hf` | second VLM, prompt-scaffolded ReAct |
| `LlamaVision` | `meta-llama/Llama-3.2-Vision` | gated; deferred to Phase 2 |
| `OllamaClient` | local Q4 | transfer-evaluation target (no gradients) |

All HF models implement `forward_with_logits(pixel_values, input_ids, **fwd_kwargs) -> logits` (gradient-friendly forward) and `generate_from_pixel_values(pixel_values, prompt, template_image, **gen_kwargs) -> text` (post-attack inference that bypasses the image-processor's pixel re-quantization).

### 2.2 Agent (`src/adversarial_reasoning/agents/medical_agent.py`)

`MedicalAgent` runs a tool-calling loop with max 8 steps:
1. Render system prompt (tool-forcing preamble: "you must invoke at least one tool before final answer").
2. Call VLM → emit text. Parse JSON `{"name": ..., "arguments": ...}` or Hermes `<tool_call>` blocks.
3. Dispatch each call against a `ToolRegistry`. Append tool result to running prompt.
4. Repeat until model emits a plain-text conclusion or hits `max_steps`.

Two entry points:
- `run(image, prompt)` — clean PIL image, full image-processor path.
- `run_with_pixel_values(pixel_values, gen_kwargs)` — adversarial inference: pixel tensor is fixed across steps; `gen_kwargs` carries model-specific extras (`image_grid_thw` for Qwen, `image_sizes` for LLaVA-Next).

### 2.3 Tools (`src/adversarial_reasoning/tools/`)

Sandboxed JSON-RPC tools, deterministic stubs:
- `query_guidelines` — keyword lookup over a small NCCN-flavored DB.
- `lookup_pubmed` — Pubmed-stub retrieval.
- `calculate_risk_score` — risk calculator (PI-RADS-like).
- `draft_report` — radiology-report scaffolding.
- `request_followup`, `escalate_to_specialist` — workflow actions.

The tool surface is small on purpose: it makes trajectory edit-distance interpretable.

### 2.4 Tasks & data (`src/adversarial_reasoning/tasks/`)

Single task in scope: `prostate_mri_workup`. Source: BHI ProstateX CV folds (`fold_{1,2,3}_X_val_3D.npy`), 8-bit slices rendered as 3-channel RGB.

Logical-split → fold mapping (`_BHI_DEFAULT_SPLIT_TO_FOLD`):
```
train → 1, dev → 2, test → 3, val → 1
```
Smokes pin `--split val` (fold 1, 5 patients). Sweeps use the same fold and a `dataset_split: { val: N }` cap.

---

## 3. Attack suite (`src/adversarial_reasoning/attacks/`)

All attacks operate on the model's **post-normalization pixel tensor** so gradients flow cleanly. ε is reported in raw 0–1 pixel space (e.g. `8/255 ≈ 0.0314`); CLIP normalization (std≈0.27) makes the effective L∞ in normalized space `≈ 0.116`. Same convention across both models — cross-model comparison stands.

| Attack | File | Loss | Notes |
|---|---|---|---|
| **PGD-L∞** | `pgd.py` | CE on benign tool-call tokens | random restart picks worst-loss δ |
| **APGD-L∞** | `apgd.py` | adaptive-step PGD | η₀=2ε, halves on plateau, heavy-ball momentum, warm restart from best iterate. Croce & Hein 2020 §3.2 checkpoints |
| **Targeted-Tool PGD** | `targeted_tool.py` | CE forcing target tool at step k | thin wrapper over `PGDAttack(targeted=True)` + `build_target_tokens()` |
| **Trajectory-Drift PGD** | `trajectory_drift.py` | `−KL(p_attack ‖ p_benign.detach())` on tool-call positions | benign logits cached once under `no_grad`; ascends KL across the *full* benign trajectory |
| C&W L2 | `cw.py` | margin-based | scaffolded only, not in current sweeps |

All four gradient attacks are wired into `runner.py::GRADIENT_MODES` and reachable from CLI.

---

## 4. Pipeline (`src/adversarial_reasoning/runner.py`)

End-to-end flow per `(model, sample, seed, attack, ε)` cell:

```
load_task(task_id, split, n)                                     # PIL + prompt
    │
    ▼
agent.run(image, prompt)                ──►  Trajectory(benign)  # clean reference
    │
    ▼
vlm.prepare_attack_inputs(image, prompt) ──► {pixel_values, input_ids,
                                              attention_mask,
                                              image_grid_thw | image_sizes}
    │
    ▼
build_attack(mode, ε, steps, target_tool, target_step_k)
    │
    ▼
attack.run(image=pixel_values, prompt_tokens, target,
           forward_kwargs=fwd_kwargs)    ──►  AttackResult(perturbed_image, loss_final, iterations)
    │
    ▼
agent.run_with_pixel_values(perturbed, gen_kwargs=...)
                                          ──►  Trajectory(attacked)
    │
    ▼
record = {benign, attacked, edit_distance_norm, ε, seed, model, attack, elapsed_s}
            └► JSONL append to runs/<exp>/records.jsonl
```

Key invariants enforced by the runner:
- `prepare_attack_inputs` must return `pixel_values` and `input_ids`; optional `image_grid_thw` (Qwen) or `image_sizes` (LLaVA) are auto-threaded into both `fwd_kwargs` (gradient pass) and `gen_kwargs` (post-attack inference).
- `build_target_tokens(vlm, target_tool)` is used for targeted_tool, `_build_trajectory_tokens(vlm, benign, prompt_input_ids)` for drift, plain `_build_target_tokens` for the rest.
- The clean PIL image is kept as `template_image` for the post-attack agent so the processor emits the right number of `<image>` placeholder tokens, then we substitute the perturbed pixel tensor into `model.generate`.

CLI: `python -m adversarial_reasoning.runner --config <yaml> --mode {noise,pgd,apgd,targeted_tool,trajectory_drift} --split {dev,val,test}`.

---

## 5. Metrics (`src/adversarial_reasoning/metrics/`)

- **`edit_distance_norm`** — Levenshtein distance between benign and attacked tool-name sequences, divided by the longer sequence's length. **Primary metric.** 0 = unchanged trajectory, 1 = total replacement.
- **`targeted_hit`** — boolean flag set when the targeted-tool attack drove `target_tool ∈ attacked.tool_sequence()`.
- **`<mode>_loss_final`**, **`<mode>_steps`** — attack-side telemetry (final adversarial loss, iteration count).
- **Wilcoxon signed-rank** across seed-paired ε levels + 95% bootstrap CIs (`metrics/stats.py`).

---

## 6. Phase 0 — sanity gates (`src/adversarial_reasoning/gates/`)

Before running attacks, three gates verify the harness is real, not a leakage artifact:

1. **`preprocessing_transfer`** — perturbations applied in normalized-pixel space must survive the processor round-trip without saturation.
2. **`noise_floor`** — uniform noise at ε=8/255 must *not* meaningfully change the trajectory (`edit_distance_norm` < small threshold).
3. **`e2e_probe`** — hand-crafted maximal δ should drive `edit_distance_norm` toward 1, confirming the loop is end-to-end gradient-effective.

All three pass on Qwen and LLaVA at fold 1.

---

## 7. Configs (`configs/`)

| Family | Smoke | Sweep |
|---|---|---|
| Baseline noise | `smoke.yaml`, `smoke_llava.yaml` | `smoke_sweep.yaml` |
| PGD-L∞ | `pgd_smoke.yaml` | (subsumed by APGD) |
| APGD-L∞ | `apgd_smoke.yaml`, `apgd_smoke_llava.yaml` | `apgd_sweep.yaml` |
| Targeted-Tool | `targeted_tool_smoke{,_llava}.yaml` | `targeted_tool_sweep.yaml` |
| Trajectory-Drift | `trajectory_drift_smoke{,_llava}.yaml` | `trajectory_drift_sweep.yaml` |

Sweeps: ε ∈ `{2, 4, 8, 16}/255`, seeds `{0, 1, 2}`, 5 patients × 4 ε × 3 seeds = 60 records each.

Static schemas: `attacks.yaml` (registers attack classes + default hyperparameters), `models.yaml`, `tasks.yaml`, `experiment.yaml`.

---

## 8. Current results

### 8.1 Smoke (n=5 patients, ε=8/255, seed=0)

| attack | qwen edit_dist | llava edit_dist |
|---|---|---|
| APGD | **0.427** | 0.297 |
| Targeted-Tool | 0.347 | **0.475** |
| Trajectory-Drift | **0.389** | 0.206 |

LLaVA is more robust to APGD/Drift but more disrupted by Targeted-Tool. Targeted-hit rate is 0% on both at L∞ ε≤16/255 — the targeted CE loss disrupts the trajectory but does not reliably invoke the chosen tool.

### 8.2 Sweep (Qwen only so far, n=60 per attack)

| attack | mean edit_dist | best ε | notes |
|---|---|---|---|
| APGD | strongest overall | 16/255 | monotonic in ε across seeds |
| Targeted-Tool | second | 16/255 | hit-rate stays 0% |
| Trajectory-Drift | third | 16/255 | KL plateaus before edit_dist saturates |
| uniform noise | floor | — | ≈ 0.1 across all ε |

LLaVA full sweeps are next.

### 8.3 Figures (`paper/figures/`)

- `attack_landscape/` — per-ε grouped bars + per-sample dots, sweep view.
- `hero_dark/` — dark-editorial publication suite (5 plots).
- `cross_model/edit_dist_grouped.png` + `per_sample_dot.png` — Qwen vs LLaVA at smoke scale, *new this session*.
- `comprehensive/` + `reasoning_flow/` — exploratory.

---

## 9. Reproducibility

```bash
# 0. one-time setup
./scripts/download_models.sh        # Qwen + LLaVA snapshots
./scripts/prepare_datasets.sh       # BHI fold tensors

# 1. Phase 0 gates
python -m adversarial_reasoning.gates.preprocessing_transfer
python -m adversarial_reasoning.gates.noise_floor
python -m adversarial_reasoning.gates.e2e_probe

# 2. smokes (n=5 each)
for atk in apgd targeted_tool trajectory_drift; do
  python -m adversarial_reasoning.runner \
    --config configs/${atk}_smoke.yaml --mode $atk --split val
  python -m adversarial_reasoning.runner \
    --config configs/${atk}_smoke_llava.yaml --mode $atk --split val
done

# 3. full sweeps
for atk in apgd targeted_tool trajectory_drift; do
  python -m adversarial_reasoning.runner \
    --config configs/${atk}_sweep.yaml --mode $atk --split val
done

# 4. figures
python scripts/compare/attacks.py --runs apgd=runs/apgd_sweep \
  targeted_tool=runs/targeted_tool_sweep \
  trajectory_drift=runs/trajectory_drift_sweep \
  --out paper/figures/attack_comparison
python scripts/make/attack_landscape.py
python scripts/make/hero_figures.py
python scripts/compare/models.py     # cross-model
```

All randomness funnels through `torch.manual_seed(seed)` + `numpy.random.seed(seed)` at the start of each `(seed, sample)` cell. Records are JSONL append-only, schema versioned implicitly by field set.

---

## 10. Limitations

- **Single dataset** (BHI ProstateX). MIMIC-CXR, BraTS were planned but cut for scope.
- **Targeted-hit rate stays 0%** at L∞ ε≤16/255 — likely needs L2 (C&W) or longer step budgets; an open question.
- **Tool semantics matter as much as token IDs.** Two trajectories with different tool *names* but the same *clinical effect* score as different by edit-distance. A semantic equivalence layer is on the wishlist.
- **Closed-API transfer** (Ollama-Q4 transfer evaluation) is wired but not yet run at sweep scale.
- **C&W L2 attack** is scaffolded but not in production sweeps.
- **No defense baselines yet.** Phase 2 will compare against published Med-VLM defenses — RobustVLM (Schlarmann et al. 2024, adversarially fine-tuned vision encoder), PromptSmooth++ (MICCAI 2025, certified robustness via randomized smoothing), and input-purification (DiffPure, JPEG re-encoding, random-resize-and-pad). The current report measures *attack* strength only and does not yet bound the residual risk after a defense layer.

---

## 11. Repo layout

```
adversarial-reasoning-attacks/
├── src/adversarial_reasoning/
│   ├── runner.py                 # CLI entry, run_gradient_attack
│   ├── agents/medical_agent.py   # ReAct loop, run / run_with_pixel_values
│   ├── attacks/                  # pgd, apgd, targeted_tool, trajectory_drift, cw
│   ├── models/                   # qwen_vl, llava, llama_vision, ollama_client
│   ├── tasks/loader.py           # BHI fold loader, _BHI_DEFAULT_SPLIT_TO_FOLD
│   ├── tools/                    # registry + deterministic stubs
│   ├── metrics/                  # trajectory edit-distance, stats
│   └── gates/                    # phase 0 sanity probes
├── configs/                       # smoke + sweep YAMLs (5 attacks × 2 models)
├── scripts/                       # figures + compare_models
├── runs/                          # JSONL records (one dir per experiment)
├── paper/figures/                 # publication-ready PNGs
└── docs/                          # ethics, limitations, reproducibility, this report
```
