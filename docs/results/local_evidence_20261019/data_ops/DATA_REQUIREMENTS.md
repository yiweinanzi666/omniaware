# Data requirements (measured 2026-10-13)

Reference repository: `yiweinanzi666/omniaware`, main commit `162115ecea8cdd9b2d1503f2d304e3066f4adb8c` (read-only clone at `D:\WWW2027\omniaware-reference`). The repository's current v2 protocol requires WebQA and OK-VQA for the main experiment; HybridQA and VideoRAG-Synth are retained generalization inputs in the data-preparation scope.

| Dataset | Repository use | Source/version | Question set | Required resources | Loader/path | Scope now |
|---|---|---|---|---|---|---|
| WebQA | `webqa_full` distractor; `webqa_open_img` secondary setting | WebQnA/WebQA main release; GitHub README points to Google Drive folder `1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ` | train+val 41,732; test 7,540; v2 test uses 4,966 val questions | train/val JSON, test JSON, 51 `imgs.7z.001`–`.051`, `imgs.lineidx`, extracted `imgs.tsv` | `WEBQA_DIR` / `paths.WEBQA_JSON`; open artifacts are separate | full raw release required; currently incomplete |
| OK-VQA | `okvqa_gs` | official OK-VQA val2014 release; repository `scripts/setup/prepare_okvqa.py` | official val 5,046; v2 reserves 500 calibration + 256 dev, remainder test | question and annotation zips, selected COCO val2014 JPEGs, GS-112K corpus | `OMNIAWARE_ASSETS/data/okvqa...`, `GS_DIR`; selected subset must retain resource closure | 10%–50% preparatory subset planned |
| HybridQA | UniversalRAG generalization path | UniversalRAG HybridQA download scripts; commit must be recorded before use | repository query set; denominator to be measured from official query file | full table rows for selected IDs, linked passages, rankings/mappings | `OMNIAWARE_UNIVERSALRAG_DIR` and `UNIVERSALRAG_RANKINGS` | 10% minimum preparatory subset planned |
| VideoRAG-Synth | UniversalRAG generalization path | UniversalRAG VideoRAG download scripts | repository query set grouped by source video | selected videos/clips, subtitles/time metadata, rankings and sampled frames | `OMNIAWARE_UNIVERSALRAG_DIR`, `UNIVERSALRAG_FRAMES` | 10% minimum preparatory subset planned |

No model download, embedding generation, training, inference, scoring, or GPU work is part of data preparation.

## Current measured status

- Local D: NTFS, 350,713,352,192 bytes total, 350,578,647,040 bytes free at start; reserve requirement (30 GiB) is currently satisfied.
- Remote target: `node04`, `/home/user/Futaoran`, user `user`; filesystem `node01:/home`, 7,999,426,920,448 bytes total, 7,584,705,675,264 bytes available, inode use 1%.
- Remote WebQA has complete-size `imgs.7z.001`–`.003`; `.004.uploading` is partial; `.005`–`.012` are partial; `.013`–`.051` are missing. Remote has complete uncompressed JSON and lineidx, plus gzip copies. No `imgs.tsv` is present.
- The local transfer currently stages gzip JSON and lineidx under `D:\WebQA`; hashes are recorded after transfer. Raw image archive is not yet complete.

This file is a plan/state record, not a completion claim. See `TASK_STATE.json` and `FINAL_DATA_REPORT.md` for current truth.
