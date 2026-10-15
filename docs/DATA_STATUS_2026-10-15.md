# Data and experiment status — 2026-10-15

This note records the assets prepared for the Omni-Aware v2 handoff. Large
assets stay outside Git; this file records their paths, counts, hashes, and
known limitations so a run can be reproduced without treating a partial
download as a complete benchmark.

## Prepared assets

The local data root is `D:\\OmniAware_Data`.

| Dataset | Measured state |
| --- | --- |
| OK-VQA | 9,009 train questions and 5,046 validation questions; 14,031/14,031 referenced COCO images; question coverage 100%. |
| GS-112K / GS-168K | 112,724 and 168,306 non-empty passages; source CSV and prepared JSONL are present. |
| DPR Wikipedia | Verified source gzip, 4,694,541,059 bytes; first 3,000,000 passages prepared. |
| HybridQA | Official WikiTables and HybridQA archives retained; 2,000/2,000 suite questions have their table and passage sources. |
| WebQA | 51/51 Hugging Face mirror volumes present; total 54,445,524,319 bytes; seven-Zip multi-volume test passed. JSON SHA-256 values are recorded in the local data-operations manifest. The extracted `imgs.tsv` is 72,234,224,104 bytes. The open pool contains 545,820 text rows and 389,750 image rows; 389,740 images decoded successfully and 10 are recorded as undecodable. |
| VideoRAG-Synth | Deliberately partial: 1,199/1,725 sampled videos (13.91% of the 8,621 candidate population), 174/178 gold videos, and 366/374 evaluation questions covered. |

WebQA mirror hashes are provenance for the mirror used to obtain the volumes;
they are not official WebQnA checksums. The archive test is the independent
integrity check for the concatenated 7z volumes.

## Reproducibility paths

The WebQA source directory is `D:\\WebQA`. A junction is available at
`assets\\downloads\\webqa`; the JSON copies used by the project are under
`assets\\data`. Set `OMNIAWARE_ASSETS=D:\\OmniAware_Data\\assets` and pass
`--webqa-dir D:\\WebQA` to `scripts.setup.prepare_webqa_open` when rebuilding
the pool or image index.

The project data manifest remains at `manifests/SUMMARY.json`. VideoRAG's
missing gold IDs are listed in that manifest and in
`manifests/videorag_stopped_20261015_142952.json`; a run must either use the
available-question file explicitly or obtain the missing videos.

## Experiment boundary

The available data are sufficient for data preparation and WebQA/OK-VQA v2
runs once the model and retrieval encoders are installed. The current machine
does not contain the Qwen/InternVL/BGE/CLIP weights, and the remote node's
initial package installation was blocked by its network until a proxy tunnel
was established. No real-model score is claimed by this data note. Any pilot
run made with the VideoRAG subset must report the reduced denominator and the
missing-question list.
