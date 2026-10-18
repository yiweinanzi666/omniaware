param(
    [ValidateRange(0,60)][int]$Seconds = 30,
    [switch]$RefreshMirrorHashes,
    [switch]$VerifyComplete
)

$ErrorActionPreference = 'Stop'
$ops = 'D:\WWW2027\data_ops'
$root = 'D:\WebQA\chunks'
$manifest = Get-Content -LiteralPath (Join-Path $ops 'WEBQA_HF_MANIFEST.json') -Raw | ConvertFrom-Json
$mirrorPath = Join-Path $ops 'WEBQA_HF_HASHES.json'
if ($RefreshMirrorHashes) {
    $entries = Invoke-RestMethod -Uri 'https://huggingface.co/api/datasets/Ancci/webqa_large/tree/main?recursive=true' -Proxy 'http://127.0.0.1:7897' -TimeoutSec 45
    $records = @($entries | Where-Object path -Match '^imgs\.7z\.\d{3}$' | ForEach-Object {
        [pscustomobject]@{ name=$_.path; bytes=$_.size; sha256=$_.lfs.oid }
    })
    if ($records.Count -ne 51) { throw 'Mirror inventory does not contain 51 chunks.' }
    foreach ($r in $records) {
        if ($r.sha256 -notmatch '^[0-9a-f]{64}$') { throw "Missing mirror SHA-256 for $($r.name)" }
        $expected = @($manifest.records | Where-Object name -EQ $r.name)
        if ($expected.Count -ne 1 -or [int64]$expected[0].content_length -ne [int64]$r.bytes) {
            throw "Mirror size does not match download manifest for $($r.name)"
        }
    }
    [ordered]@{ measured_at=(Get-Date -Format o); source='hf:Ancci/webqa_large'; checksum_authority='mirror Git LFS SHA-256; not an official WebQA checksum'; records=$records } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $mirrorPath -Encoding UTF8
}

function Get-Snapshot {
    $rows = @(foreach ($r in $manifest.records) {
        $final = Join-Path $root $r.name
        $part = "$final.part"
        $item = $null
        $state = 'missing'
        if (Test-Path -LiteralPath $final) {
            $item = Get-Item -LiteralPath $final
            $state = if ($item.Length -eq [int64]$r.content_length) { 'downloaded_unverified' } else { 'size_mismatch' }
        } elseif (Test-Path -LiteralPath $part) {
            $item = Get-Item -LiteralPath $part
            $state = 'downloading'
        }
        $prefixBytes = if($item){[int64]$item.Length}else{0}
        $stagedBytes = [int64]0
        $stageDir = Join-Path (Join-Path $root '.ranges') $r.name
        if ($state -ne 'downloaded_unverified' -and (Test-Path -LiteralPath $stageDir)) {
            $uniqueRanges = @{}
            foreach ($piece in Get-ChildItem -LiteralPath $stageDir -File) {
                if ($piece.Name -match '^(\d+)-(\d+)\.(range|ready)$') {
                    $startOffset = [int64]$Matches[1]
                    $endOffset = [int64]$Matches[2]
                    $receivedBytes = [math]::Min([int64]$piece.Length, $endOffset-$startOffset)
                    $uniqueBytes = [math]::Max(0,$startOffset+$receivedBytes-[math]::Max($prefixBytes,$startOffset))
                    $key = "$startOffset-$endOffset"
                    if (!$uniqueRanges.ContainsKey($key) -or $uniqueBytes -gt $uniqueRanges[$key]) {
                        $uniqueRanges[$key] = $uniqueBytes
                    }
                }
            }
            foreach ($uniqueBytes in $uniqueRanges.Values) { $stagedBytes += [int64]$uniqueBytes }
        }
        [pscustomobject]@{ name=$r.name; bytes=($prefixBytes+$stagedBytes); prefix_bytes=$prefixBytes; staged_bytes=$stagedBytes; expected=[int64]$r.content_length; state=$state }
    })
    [pscustomobject]@{ bytes=[int64]($rows | Measure-Object bytes -Sum).Sum; rows=$rows }
}

$before = Get-Snapshot
$timer = [Diagnostics.Stopwatch]::StartNew()
if ($Seconds -gt 0) { Start-Sleep -Seconds $Seconds }
$after = Get-Snapshot
$timer.Stop()
$rate = if ($Seconds -gt 0) { ($after.bytes-$before.bytes)/$timer.Elapsed.TotalSeconds/1MB } else { $null }
$total = [int64]($after.rows | Measure-Object expected -Sum).Sum
$processes = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -match 'webqa_drive_downloader\.py' })
$status = [ordered]@{
    measured_at=(Get-Date -Format o); source='hf:Ancci/webqa_large'; source_manifest=(Join-Path $ops 'WEBQA_HF_MANIFEST.json')
    chunks_total=$after.rows.Count; chunks_partial=@($after.rows | Where-Object state -EQ 'downloading').Count
    chunks_complete=@($after.rows | Where-Object state -EQ 'downloaded_unverified').Count
    bytes_landed=$after.bytes; bytes_contiguous=[int64]($after.rows | Measure-Object prefix_bytes -Sum).Sum
    bytes_staged=[int64]($after.rows | Measure-Object staged_bytes -Sum).Sum
    bytes_target=$total; percent=[math]::Round(100*$after.bytes/$total,3)
    recent_rate_mib_s=$rate; measurement_seconds=$timer.Elapsed.TotalSeconds
    estimated_remaining_hours=if($rate -gt 0){[math]::Round(($total-$after.bytes)/1MB/$rate/3600,2)}else{$null}
    process_ids=@($processes | Select-Object -ExpandProperty ProcessId); proxy_url='http://127.0.0.1:7897'
    resume_mode=if(Test-Path -LiteralPath (Join-Path $root '.ranges')){'continuous .part prefix plus persistent disjoint ranges'}else{'continuous .part prefix'}
    disk_free_bytes=[int64](Get-PSDrive D).Free
    other_datasets='user downloads other datasets; current download scope is WebQA'
}
$status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ops 'WEBQA_DOWNLOAD_STATUS.json') -Encoding UTF8
$status | ConvertTo-Json -Depth 5 -Compress | Add-Content -LiteralPath (Join-Path $ops 'WEBQA_MEASUREMENTS.jsonl') -Encoding UTF8

if ($VerifyComplete) {
    $mirror = Get-Content -LiteralPath $mirrorPath -Raw | ConvertFrom-Json
    $verified = @(foreach ($row in $after.rows | Where-Object state -EQ 'downloaded_unverified') {
        $file = Join-Path $root $row.name
        $initial = Get-Item -LiteralPath $file
        $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        $current = Get-Item -LiteralPath $file
        $reference = @($mirror.records | Where-Object name -EQ $row.name)
        $stable = $initial.Length -eq $current.Length -and $initial.LastWriteTimeUtc -eq $current.LastWriteTimeUtc
        [pscustomobject]@{ name=$row.name; bytes=$current.Length; sha256=$hash; stable=$stable; mirror_sha256_matches=($reference.Count -eq 1 -and $hash -eq $reference[0].sha256 -and $stable) }
    })
    [ordered]@{ measured_at=(Get-Date -Format o); checksum_authority=$mirror.checksum_authority; archive_test='NOT_RUN'; records=$verified } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ops 'WEBQA_COMPLETED_HASHES.json') -Encoding UTF8
}
$status | ConvertTo-Json -Depth 5
