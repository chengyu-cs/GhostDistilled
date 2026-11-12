# Supplementary Material

Open recipe and run records for the paper *Ghost, Distilled: An Open Recipe for
Real-Person Persona Simulation from Twelve Years of Public Utterances*
(camera-ready version, six pages).

Everything under `results/` is archived output of the runs cited by the paper's
tables and inline numbers. Nothing there was regenerated for this package:
transcripts and scores are the originals, renamed to functional names, with
machine-internal path echoes stripped (all verdict lines and measurement values
byte-identical to the logged runs of October 2026).

## Content tree

```
corpus/                          the released corpus + how each layer was counted
  huangxiaoyun-public-corpus/    the public corpus package itself (data/, analysis/, docs/, index/, refined/)
  make_funnel_counts.py          corpus funnel counting script (Table I)
  build_public_package.py        assembles the public package from the master pool
style_anchors/                   the artist's own measured baselines (Table II)
  bootstrap_anchor_ci.py         register anchors with bootstrap 95% CIs
  register_statistics.py         channel x era statistics
evaluation_protocol/             the 100-turn deterministic protocol
  answer_key.json                question-to-ground-truth map + sass lexicon (the ruler)
  protocol_turns.json            the 100 scripted turns
  score_transcripts.py           deterministic scorer (lyric/memory/style/policy)
  conversation_metrics.py        length-variance / sass computation
  run_teacher_stack.py           run the protocol against the served teacher
  run_zeroshot_baseline.py       zero-shot baseline (same LLM, one-line persona)
  run_retrieval_fewshot.py       retrieval few-shot baseline (corpus top-5)
  run_student_endpoint.py        run against a student endpoint (hosted or local)
  judge_30turn_protocol.py       the 30-turn judged protocol (strict + fan personas)
  boundary_probes_teacher.py     20 adversarial probes against the teacher
  boundary_probes_student.py     same probes against a student
teacher_pipeline/                the eight-module prompt-engineered teacher
  (agent.py + prompt/policy/lexicon/memory/memory2/retrieval/kb_retrieval/
   lyrics_store/style_anchor/hybrid/bon/canned_text/facts)
distillation/                    distillation data + training configuration
  build_distillation_dialogues.py   generate teacher dialogues through the real pipeline
  prepare_distillation_mix.py       filter, prefix-expand, blend with the base set
  prepare_sft_base_mix.py           the plain-SFT base mix builder
  train_student_qlora.py            4-bit QLoRA config (rank 16, 2 epochs, lr 1e-4)
  serve_student_vllm.py             local serving of a trained student
  distill_dialogues_{train,val}.jsonl   the 389 curated dialogue groups, expanded
  distill_mix_{train,val}.jsonl        dialogue groups + 1,186-sample base blend
  sft_base_{train,val}.jsonl           plain-SFT training set (base samples only)
  dose_{100,200,389,389only}groups_{train,val}.jsonl   dose-response arms
results/                         archived run records cited by the paper
  corpus_funnel/                 funnel counts (Table I)
  register_anchors/              anchor CIs + register statistics (Table II)
  main_100turn/<system>/         the seven systems of Table III (+ temperature-1.0 reruns)
  dose_response/                 the four dose arms (Sec. IV-B)
  module_ablation/               single-module removals (Sec. IV-B)
  serving_tiers/                 single-pass / parallel-pair / auto-routed (Sec. IV-B)
  boundary_probes/               20 adversarial probes per system (Table IV)
  judge_30turn/                  judged-protocol runs (teacher/students/27B line/judge-in-loop/top_p A-B)
  judge_stability/               five repeat scorings of one transcript
```

Each results directory carries a `NOTES.md` (what was run, on which model,
verdict lines quoted verbatim from the archived outputs).

## Models

| Role | Model | Sampling |
|---|---|---|
| Teacher | general-purpose chat LLM behind the eight-module stack | T=1.0 |
| Zero-shot / retrieval baselines | same general-purpose LLM, one-line persona (+ corpus top-5) | T=1.0 |
| Distilled student (14B) | Qwen2.5-14B-Instruct + QLoRA, checkpoint `qwen3-14b-1dc235b6bcee` | T=0.85 (reruns at 1.0) |
| Plain-SFT student (14B) | Qwen2.5-14B-Instruct + SFT, checkpoint `qwen3-14b-35b623e73d9b` | T=0.85 (reruns at 1.0) |
| Distilled student (27B) | parallel 27B line, versions 1-4, locally served (vGPU) | tuned per version |
| Character base | CharacterGLM-6B, zero-shot with the students' persona prompt | T=0.85 |
| Judge (30-turn) | general-purpose LLM, strict + fan personas | T=0.2 |

## Run directory -> paper claim map

| Paper claim | Where |
|---|---|
| Table I funnel counts | `results/corpus_funnel/`, `corpus/make_funnel_counts.py` |
| Table II register anchors + CIs | `results/register_anchors/`, `style_anchors/` |
| Table III, all seven rows | `results/main_100turn/<system>/` (each `scores.json` has the per-row numbers) |
| Table III note: T=1.0 reruns | `results/main_100turn/*/temperature_1.0/` |
| Table III note: vGPU latencies | per-run `run.log` in the 27B and character-base directories |
| Table IV boundary retention | `results/boundary_probes/` (`scores.json` per system) |
| Judge stability 72x5 | `results/judge_stability/repeat_scores.json` |
| 30-turn scores 72/78, 74/76, 72/72 | `results/judge_30turn/{teacher_stack,distilled_student_14b,sft_student_14b}/` |
| 27B line 62->68->72->78 | `results/judge_30turn/student_27b_version{1..4_final}/` |
| Rule 1: three judge-in-loop variants | `results/judge_30turn/judge_in_loop_{best_of_n,cascade,paraphrase_fix}/` |
| Rule 2: top_p A/B, echo 4.33->1.0 | `results/judge_30turn/sampling_top_p_ab/` (run metrics carry echo_turns per run) |
| Module ablations | `results/module_ablation/` |
| Dose-response (47->72->89) | `results/dose_response/` |
| Serving-tier trade | `results/serving_tiers/` |
| Distillation blend (389 + 1,186) | `distillation/` (line counts in each file) |

## Reproduction

```bash
# corpus funnel (stages computable from the released pool)
python3 corpus/make_funnel_counts.py

# register anchors with bootstrap CIs (Table II)
python3 style_anchors/bootstrap_anchor_ci.py

# rebuild the protocol turns, then score any archived transcript
python3 -c "import sys; sys.path.insert(0,'evaluation_protocol'); import score_transcripts as s; s.build_100()"
python3 evaluation_protocol/run_zeroshot_baseline.py --tag zeroshot   # needs DEEPSEEK_API_KEY

# train the student (4-bit QLoRA, one 32GB GPU, ~95 min for the 14B)
python3 distillation/train_student_qlora.py
```

Two inputs are intentionally not shipped and gate exact re-derivation, both
stated in the paper: (1) the pre-filter corpus pools behind the funnel's
cleaning/safety stages contain material removed by the public-safety filter, so
they cannot be redistributed — the released master pool, the counting script,
and the archived counts define everything from the released layer onward;
(2) the private lyric store (106 songs) backs the teacher's retrieval and the
answer key's ground truth and is never published.

## Notes on scope

- `teacher_pipeline/` is the serving stack as deployed; it additionally contains
  a fan-facing quiz game and session conveniences that the paper does not claim;
  the eight modules the paper describes are the persona/policy/prompt stack,
  lyric retrieval with output-side anti-recitation, four-level memory, the
  AI-flavor linter, and the behavioral action pool.
- The corpus package (`corpus/huangxiaoyun-public-corpus/`) is the public
  release layer, CC BY-NC-SA with the takedown commitment in its LICENSE.
