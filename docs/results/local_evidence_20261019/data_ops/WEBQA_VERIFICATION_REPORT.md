# WebQA verification report (2026-10-13)

This report records the measured state on `node04` and is not a completion claim.

- Target: `node04:/home/user/Futaoran/WebQA`; identity verified with `hostname`, `whoami`, and `id`.
- Official source: GitHub `WebQnA/WebQA` README links Google Drive folder `1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ`; its `WebQA_imgs_7z_chunks` subfolder is `19ApkbD5w0I5sV1IeQ9EofJRyAjKnA7tb`.
- Official listing shows 51 binary files `imgs.7z.001` through `imgs.7z.051`, each displayed as 1 GB; exact byte sizes and hashes are not exposed in the listing.
- Present remote archive state: `.001`–`.003` are exactly 1,073,741,824 bytes; `.004.uploading` is 220,594,176 bytes; `.005`–`.012` are partial; `.013`–`.051` are absent. No complete archive test is possible.
- `WebQA_train_val.json`: 41,732 records, 661,922,314 bytes, SHA-256 `7ff6ba4aae749dbdd6400c69251e97a785e689039206d91d7541bec15eab3dbe`.
- `WebQA_test.json`: 7,540 records, 116,724,654 bytes, SHA-256 `a15cea11c0fbd49c425fcf9a01fc47da4f5f2674e7481751261160fd88b233db`.
- `imgs.lineidx`: 4,616,775 bytes, SHA-256 `1772b8ac88563953f36da85c52c390db90d25c128c7593643a97c8ac7468d46f`.
- `imgs.tsv`: absent; JSON-to-image reference closure, line index mapping, Base64 validation, pixel decode, and 7z integrity test are `NOT_RUN`.

Status: `WEBQA_DATA_READY = FALSE` until all 51 official parts are obtained, tested as one 7z archive, extracted to `imgs.tsv`, and validated.
