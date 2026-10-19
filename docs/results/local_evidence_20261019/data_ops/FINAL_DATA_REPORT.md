# Final data report (live, 2026-10-13)

Status is intentionally conservative: the requested data deployment is **not complete**.

| Dataset | Scope | Samples/denominator | Raw download GiB | Remote usage | Validation | Remaining |
|---|---|---:|---:|---|---|---|
| WebQA | full official release | train+val 41,732; test 7,540 | unknown until official object sizes are measured | `/home/user/Futaoran/WebQA`; 001–003 full-size, 004–012 partial, 013–051 missing | JSON/lineidx hashes measured; archive and TSV not validated | recover all 51 parts, local 7z test, transfer and extract `imgs.tsv`, validate all mappings |
| OK-VQA | preparatory subset | not selected | not measured | not deployed | not run | freeze IDs and fetch linked annotations/COCO images/GS corpus |
| HybridQA | preparatory subset | not selected | not measured | not deployed | not run | resolve UniversalRAG source and linked table/passages |
| VideoRAG-Synth | preparatory subset | not selected | not measured | not deployed | not run | resolve query/video inventory, subtitles, clips and frames |

`WEBQA_DATA_READY = FALSE`
`WEBQA_PIXEL_DECODE_CHECK = NOT_RUN`
`WEBQA_SOURCE_DEFECTS = UNRESOLVED`
`WEBQA_CURRENT_LOADER_READY = FALSE`
`OTHER_SUBSETS_READY = FALSE`
`DOWNLOAD_BUDGET_STATUS = UNRESOLVED`
`DISK_SAFETY_STATUS = PASS`
`ALL_REQUESTED_DATA_READY = FALSE`
`FORMAL_EXPERIMENT_READY = NOT_ASSESSED`

The remote L20 GPUs were observed separately and are not used by this preparation task.
