# Data and experiment status — 2026-10-15

Large data stay outside Git; this file records measured readiness.

| Dataset | State |
| --- | --- |
| OK-VQA | 9,009 train + 5,046 val; 14,031/14,031 COCO images. |
| GS corpora | GS-112K 112,724 passages; GS-168K 168,306 passages. |
| DPR Wikipedia | Verified gzip; first 3,000,000 passages prepared. |
| HybridQA | 2,000/2,000 suite questions have table and passage sources. |
| WebQA | 51/51 mirror volumes, 54,445,524,319 bytes; 7z test passed; extracted imgs.tsv 72,234,224,104 bytes; pool 545,820 text and 389,750 image rows; 10 images undecodable. |
| VideoRAG-Synth | Partial: 1,199/1,725 sampled videos, 174/178 gold videos, 366/374 questions covered. |

WebQA mirror hashes are provenance, not official WebQnA checksums. VideoRAG runs must use the available-question file or obtain missing videos and report the reduced denominator.

The local data root is `D:\\OmniAware_Data`; WebQA source is `D:\\WebQA`. Set `OMNIAWARE_ASSETS` to the asset root and rebuild open retrieval with `scripts.setup.prepare_webqa_open` when encoder/model assets are installed. No real-model score is claimed here until Qwen/InternVL/BGE/CLIP weights are present.