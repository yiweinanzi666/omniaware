> **LEGACY.** This directory documents the first experiment-host pool (`build_global_webqa_pool.py`, which does not run from this repository). The paper's pool numbers come from `scripts/setup/prepare_webqa_open.py pool` (`pool_stats.json`), which builds a different text key. Do not mix the two outputs.

# Phase1 global retrieval assets

This is a newly reconstructed public WebQA corpus protocol. It does not establish recovery of the historical paper corpus or row order. `GLOBAL_POOL_MANIFEST.json` binds 544489 exact-text documents and 389750 image IDs; original fact aliases remain available. No question/answer labels are used by the encoding jobs.

## Active work

- `encode_global_text.py`: BGE-large-en-v1.5 CPU fp32, two threads, exact body text, CLS/L2, length512, batch4. Full target544489. Fixed plan and per-shard receipts in `bge_text_cpu/`.
- `fetch_global_images.py`: pinned mirror download by image ID. Existing validated local pixels and downloaded pixels remain separate; complete fetch does not imply all original TSV pixels were recovered.
- `prepare_clip_assets.py` and `encode_clip_assets.py`: fixed snapshots of available, verified pixels, CLIP ViT-L/14 CPU fp32/SDPA, two threads, RGB and official resize/crop, batch4, normalized768-dimensional vectors. First snapshot `clip_images_cpu_wave0/assets.jsonl` contains309793 images. It adds four PNG metadata repairs without changing image data chunks; it does not use the three known truncated JPEGs. It is a preparation batch, not an evaluation subset or completed global index.

`CLIP_INDEPENDENT_VALIDATION.json` records128 real vectors, unit-norm/finite/ID checks, native-forward cosine error1.49e-8, and resume with zero model loads/new shards. The worker has resumed this checkpoint. Read actual process identity and fresh progress before taking action; launch receipts are historical observations.

## Continue without duplication

From project root, resume the existing CLIP snapshot with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 experiments/eab_first_token/cache/runtime/bin/python -u -B -W ignore -m experiments.eab_first_token.runs.phase1_dense_retrieval_codex_20261006.encode_clip_assets --output experiments/eab_first_token/runs/phase1_dense_retrieval_codex_20261006/clip_images_cpu_wave0
```

After later downloads, `prepare_clip_assets --output <new-wave-directory> --previous <all-earlier-assets.jsonl>` freezes only previously unseen image IDs; it refuses changed previously recorded pixels. Include both `clip_images_cpu_wave0/assets.jsonl` and `clip_images_public_revision_30172755/assets.jsonl` among the previous manifests. Encode the next bulk wave after the existing CLIP worker finishes, using the same two-thread resource allocation. Do not start a duplicate worker; the output lock also rejects concurrent writers.

The image manifest snapshots a complete prefix of the download receipt stream. Each embedding shard binds the source rows, plan and vector array SHA. Completed shards are hash-verified and reused. Pixel SHA and dimensions are checked immediately before strict decoding. An array lacking its completion receipt requires inspection, not silent recomputation. Encoding timing includes hashes, image decoding, preprocessing and image forward; load time and invocation wall time are also recorded. The one native-forward convention check uses synthetic text and is not task scoring.

## Remaining

`build_dense_index.py` builds exact FAISS `IndexFlatIP` indices in the isolated `runtime_dense_20261006` environment. It verifies source rows, receipt and array hashes, vector norms, CLIP ID order, unique IDs across image waves, then reconstructs every serialized vector to check row alignment. Full mode requires completed encoding receipts. `--pilot-shards` explicitly produces a partial engineering index; a complete CLIP snapshot is still marked incomplete globally unless every389750 distinct in-range image ID is present.

`FAISS_REAL_PREFIX_VALIDATION.json` records two1024-vector real indices and256 query checks per modality: all top-five sets equal brute-force cosine, maximum score error0, zero rebuild/byte changes on reentry. It also records rejection of incomplete full inputs, wrong row receipts, swapped image IDs, nonfinite vectors and overlapping waves. The small index is not an accuracy or global-retrieval result.

After BGE encoding completes, build the full text index with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 experiments/eab_first_token/cache/runtime_dense_20261006/bin/python -B -m experiments.eab_first_token.runs.phase1_dense_retrieval_codex_20261006.build_dense_index --kind bge --encodings experiments/eab_first_token/runs/phase1_dense_retrieval_codex_20261006/bge_text_cpu --output experiments/eab_first_token/runs/phase1_dense_retrieval_codex_20261006/faiss_bge_full
```

For CLIP use `--kind clip` with all completed nonoverlapping encoding directories after `--encodings`; preserve the explicit availability/version record before freezing the final retrieval protocol. Raw cosine values from BGE and CLIP are not automatically comparable and are not fused by this builder.

`encode_webqa_queries.py` has encoded all 256 provisional WebQA Phase1 train questions with the same frozen BGE/CLIP assets on CPU. Exact question text is preserved; BGE adds its retrieval instruction (512 tokens), CLIP uses the question directly (77 tokens). Two CLIP inputs reach truncation; no BGE input does. `WEBQA_QUERY_INDEPENDENT_VALIDATION.json` binds the arrays and confirms finite/unit vectors, ID alignment, numerical checks, and completion reentry with zero model loads and unchanged bytes. These are preparation artifacts, not retrieval/accuracy results. The cohort still needs its final consumption check; the score fusion rule is now frozen separately in `DENSE_ECDF_PROTOCOL.md`.

`PUBLIC_IMAGE_REVISION_30172755_RECEIPT.json` preserves the earlier adoption of the authentic2026-09-28 public crop231×320. The later official-loader check found that the original TSV payload equals the saved2008 archive and decodes to231×324. `ORIGINAL_IMAGE_30172755_INDEPENDENT_VALIDATION.json` verifies its pixels, CLIP/Qwen input tensors and completed vector. Use `clip_images_cpu_original_30172755` for the formal index; retain the public-crop artifact as history and do not include both versions of the same ID.

Complete all text/image encoding and build exact FAISS ID-to-vector indices. Four truncated original JPEGs, including30172755, are now available through the official tolerant PIL loading convention, materialized separately with exact RGB and processor-tensor equality; missing compressed bytes were not reconstructed. All389750 image IDs have prepared assets. See `DENSE_PIPELINE_READINESS.md` for the correction and four required image snapshots. Full retrieval, calibration, initial U and the final Phase1 protocol remain unfinished. Preserve original and reconstructed protocol identities separately, and apply the same final corpus to every comparison arm.

## Complete-source budget preparation

`audit_phase1_coverage.py` computes the complete physical prompt length for every frozen order and arm from a label-free source manifest. It takes the maximum as a common allowed input cap, prepares the existing round017 executor, checks that every source and planned copy remains, and twice renders the longest static and replication prompts with the real processor. It never enqueues generation. This revised comparison keeps the same sources; replication still consumes more actual tokens and compute.

`BUDGET_COVERAGE.md` and `coverage_{webqa,okvqa}_smoke/audit.json` record the four existing engineering questions. At1536, one OK-VQA question loses a source in all38 orders; one WebQA question loses a replica in all38 orders. Complete-prompt caps1600(WebQA) and1788(OK-VQA) pass, but these are sample-specific values. Recompute on the entire final manifest before freezing Phase1; neither changing the cohort nor selecting a cap by accuracy is part of this rule. The historical1536 smoke and its reserved30 model calls are unchanged. The new prepared directories are CPU audit artifacts, not additional model jobs.

## Mixed train calibration queries

`prepare_webqa_calibration.py` freezes500 questions from the existing train calibration pools:242 text-category and258 image-category questions, using the public train proportions and deterministic hash order. IDs and normalized questions are disjoint from the Phase1 train256. Answer/support labels and entropy values do not select the cohort. The same500 queries go through both encoders; category is not a routing oracle.

`encode_calibration_queries.py` has completed500 BGE query vectors and500 CLIP query vectors on CPU, with zero truncated inputs. `CALIBRATION_PREPARATION_INDEPENDENT_VALIDATION.json` verifies identity, finite/unit vectors, scalar/batch agreement, and reentry with zero model loads and byte changes. The original train256 query arrays remain unchanged.

`CALIBRATION500_READINESS.md` records the remaining work.242 historical initial observations have matching prompt, tokenizer/model-asset bindings and observer semantics and are reused. The other258 historical prompts/tokens match, but their execution did not record model-weight hashes; their old observations remain intact and fresh observations are pending under the Phase1 binding. Median/IQR are not frozen from a partial sample. Both complete global indices are still required before fitting the new retrieval score mappings; the old caption/BGE and question-local ECDFs are not reused as BGE/CLIP global calibration.

Reproduce from the project root, with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false`:

```bash
experiments/eab_first_token/cache/runtime/bin/python -B -m experiments.eab_first_token.runs.phase1_dense_retrieval_codex_20261006.prepare_webqa_calibration
experiments/eab_first_token/cache/runtime/bin/python -B -m experiments.eab_first_token.runs.phase1_dense_retrieval_codex_20261006.encode_calibration_queries
```

## Query-to-source search connection

`search_dense_queries.py` writes per-modality top-five cosine scores with source IDs and index rows. It binds index bytes, ID order, query arrays, encoder assets and completion receipts. Default mode requires the complete text corpus or all available images with the documented JPEG gaps; `--engineering-prefix` explicitly marks partial-index checks. Completed retrieval reentry verifies the saved output and performs no new search.

`DENSE_SEARCH_INDEPENDENT_VALIDATION.json` verifies all756 queries per modality against the original1024-vector encoding fragments: all top-five sets and source-ID rows match, maximum cosine error0. Four real mismatch cases are rejected. These checks use no labels or model calls. See `DENSE_SEARCH_READINESS.md` for the full-index command and remaining calibration steps; the engineering retrievals are not global retrieval or accuracy results.

## Initial observation runner

`observe_phase1_initial.py` prepares, checks and resumes the258 missing calibration plus256 cohort observations, reusing the242 verified historical calibration records. It calls the existing question-only entropy implementation and attempt recorder. Completed observations are bound to their task and attempt; failed/interrupted prefills and model loads retain their costs or explicit unknown times. Full500 median/IQR fitting happens only after all inputs complete.

`INITIAL_OBSERVATION_INDEPENDENT_VALIDATION.json` records514 exact prompt/token/mask matches to the historical path, six tiny-Qwen CPU tests, and preparation reentry without model loading or artifact changes. Actual8B observations are pending; the task has not taken a device slot. See `INITIAL_OBSERVATION_READINESS.md` for the command and cost boundaries.

## Calibrated source selection and reader metadata

`calibrate_dense_relevance.py` fits the frozen per-modality top5 ECDFs and selects five sources. `join_dense_source_metadata.py` adds exact text, every normalized unique caption, and the same image bytes indexed by CLIP. It requires complete global indices for formal mode and keeps U absent until actual initial observations complete. `DENSE_PIPELINE_READINESS.md` records4 CPU regressions,2560 independent percentile checks,1280 source joins and four actual reader-input renders using engineering-prefix retrievals. These checks do not supply formal calibration or accuracy results.
