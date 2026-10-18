# WebQA download recovery, 2026-10-15

The download scope is WebQA only. The user is obtaining other datasets separately.
The target remains the complete 51-volume image archive, followed by archive testing,
upload to node04, extraction and validation. Download completion is not data readiness.

## Current downloader

Entry point: `webqa_drive_downloader.py`, using `--parallel-ranges` and
`webqa_range_resume.py`. The existing contiguous prefixes are still
`D:\WebQA\chunks\imgs.7z.NNN.part`. Disjoint pieces are stored durably under
`D:\WebQA\chunks\.ranges\imgs.7z.NNN\`.

- `plan.json` fixes the source URL, object size, prefix base and range size.
- `START-END.range` contains a partial range (END is exclusive).
- `START-END.ready` is a complete range, closed and flushed before promotion.
- Only the per-file commit lock appends ready ranges in order to the archive prefix.
- Pieces are removed only after their contents are flushed into the prefix.
- An interrupted append is recovered by checking overlapping bytes before appending.
- A final archive filename is created only at the exact manifest size.
- Completed-volume SHA-256 must still match the mirror inventory; the whole archive
  still requires `7z t` when all 51 volumes are complete.

Do not delete the `.ranges` directories, truncate prefixes, or restore an old
downloader that ignores the staged pieces. The current entry point automatically
selects range recovery whenever range plans exist. It takes an exclusive output
lock to reject a second downloader. Still check the actual Windows processes before
starting anything: an older downloader may not implement the lock.

`measure_webqa.ps1` counts the contiguous prefix plus unique staged bytes beyond
that prefix, avoiding double counting. `bytes_contiguous` and `bytes_staged` are
reported separately; only a final file at the expected size counts as a complete volume.

## Evidence and tuning

The previous downloader PID 9144 had one connection per remaining volume. Its 48
configured workers fell to about 22 actual downloads as volumes finished. It was
stopped only after the replacement passed local HTTP tests. All 51 existing file
boundary checks passed after switching; the takeover snapshot is
`WEBQA_PARALLEL_TAKEOVER.json`.

`test_webqa_range_resume.py` passed five tests: short responses during concurrent
download, resume from partial/completed staged ranges, interrupted append recovery,
reject invalid HTTP/Range responses without changing existing bytes, and exclusive
writer locking. Test data are local synthetic bytes, not duplicate dataset downloads.

The initial 48-connection trial with a 35-second socket timeout was slower (0.093
and 0.212 MiB/s in measured windows). A 24-connection trial with a 120-second socket
timeout measured 0.396 and 0.326 MiB/s. The active configuration subsequently became
48 connections with a 120-second socket timeout. Its first minute was slow, but
the later 15:23:52 to 15:30:12 window averaged about 0.478 MiB/s. Single-minute
samples are noisy; use longer byte-delta windows when estimating completion.
Consult `WEBQA_DOWNLOAD_STATUS.json` and current process arguments for the latest
observed rate and PID. The active run at 15:30 was PID 11216, using
`WEBQA_PARALLEL_48_120.stdout.log` and `WEBQA_PARALLEL_48_120.stderr.log`.

By 15:20, 35 completed volumes had SHA-256 values matching the HF mirror.
`WEBQA_COMPLETED_HASHES.json` contains the actual per-volume results. These are
mirror hashes, not official WebQA hashes. Archive testing has not yet run.

The proxy currently listens at `http://127.0.0.1:7897`. The old selector name
`🔰 选择节点` is no longer present in the current proxy configuration. Do not use
the old `_set_clash_node.ps1` command uncritically or alter the system proxy.

## Logs and restart

Inspect the unique Python downloader process, then its current stdout/stderr log.
Range retries and completions are also appended to
`D:\WebQA\chunks\logs\parallel_ranges.jsonl`; major events also go to
`WEBQA_HF_PROGRESS.jsonl`. A live process plus files growing is the evidence of
progress; neither an old PID nor a saved status file proves that it is still running.

After confirming no downloader is running, use the bundled Python executable with:

```
D:\WWW2027\data_ops\webqa_drive_downloader.py --source hf --workers 48 --range-size 4194304 --proxy-url http://127.0.0.1:7897 --manifest D:\WWW2027\data_ops\WEBQA_HF_MANIFEST.json --progress D:\WWW2027\data_ops\WEBQA_HF_PROGRESS.jsonl --parallel-ranges
```

Keep the background window hidden, preserve previous logs, and measure actual
progress after starting. Never start a second downloader or put SSH credentials
in scripts, commands, files, environment variables, or reports.

## Archive test tool

`D:\WWW2027\data_ops\tools\7zr.exe` is the official standalone 7-Zip (r) 26.04
console tool, obtained over verified HTTPS from
`https://www.7-zip.org/a/7zr.exe`. Its size is 602624 bytes and SHA-256 is
`256feca8e274e5da655e2a284fabafd9f554365eb164862089dacd4e8276d282`.
The `i` command succeeded and lists support for 7z, split volumes and LZMA2.

Once all 51 final volumes are stable and their mirror SHA-256 values pass, run
`7zr.exe t D:\WebQA\chunks\imgs.7z.001`, preserving its complete log and actual
exit code. Do not claim an archive test passed merely because the tool is ready.
The test has not run as of 2026-10-15 16:33; 49 volumes were complete and matched
mirror SHA-256 values then, while volumes 049 and 050 were still downloading.
