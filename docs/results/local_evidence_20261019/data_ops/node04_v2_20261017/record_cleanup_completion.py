"""Preserve completed cleanup evidence and update the preparation report."""
import collections
import hashlib
import json
import pathlib
import shutil

ops = pathlib.Path('D:/WWW2027/data_ops/node04_v2_20261017')
repo = ops / 'review_7ccf676'
evidence = repo / 'docs/results/20261017_node04_preparation_evidence'
receipts = sorted(ops.glob('cleanup_executed_*.json'))
assert len(receipts) == 1, receipts
receipt_path = receipts[0]
receipt = json.loads(receipt_path.read_text(encoding='utf8'))
assert receipt['execute'] is True
journal = pathlib.Path(receipt['journal'])
counts = collections.Counter()
totals = collections.Counter()
sha = hashlib.sha256()
with journal.open('rb') as source:
    for line in source:
        sha.update(line)
        row = json.loads(line)
        counts[row['action']] += 1
        if row['action'] == 'deleted':
            totals[row['g']] += row['bytes']
assert counts['deleted'] == receipt['deleted_files']
assert sum(counts.values()) == receipt['eligible_files_checked']
assert counts['retained'] == receipt['retained_after_verification']
assert sum(totals.values()) == receipt['deleted_logical_bytes']
comparison = json.loads((ops / 'cleanup_comparison_1725.json').read_text(encoding='utf8'))
for item in comparison['retained']:
    row = item['local']
    path = pathlib.Path('D:/WWW2027') / comparison['groups'][row['g']][0] / row['r']
    assert path.is_file(), path
    assert path.stat().st_size == row['b'], path
audit = {
    'receipt': receipt_path.name,
    'receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    'journal_filename': journal.name,
    'journal_sha256': sha.hexdigest(),
    'journal_action_counts': dict(counts),
    'deleted_bytes_by_group': dict(totals),
    'all_26_excluded_files_still_exist_with_original_size': True,
    'github_pre_cleanup_gate': json.loads((ops / 'github_push_verified.json').read_text()),
    'free_space_delta_bytes': receipt['free_bytes_after'] - receipt['free_bytes_before'],
}
shutil.copyfile(receipt_path, evidence / receipt_path.name)
shutil.copyfile(ops / 'github_push_verified.json', evidence / 'github_push_verified.json')
(evidence / 'cleanup_completion_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
report = repo / 'docs/results/20261017_node04_preparation.md'
text = report.read_text(encoding='utf8')
assert '## GitHub 推送与实际清理完成' not in text
text += f'''
## GitHub 推送与实际清理完成

Git Credential Manager 设备授权成功后，准备报告分支 `results/node04-preparation-20261017` 已推送至 `https://github.com/yiweinanzi666/omniaware`；推送退出码 0，随后 `git ls-remote` 核实远端为 `681dea00171fc21f72e85ee77007eb90a767ba94`。通过这道条件后才执行本地清理。本节随后的提交用于保留实际清理回执。

- 清理开始：{receipt['start']}；结束：{receipt['end']}。
- 逐文件检查 {receipt['eligible_files_checked']:,} 个路径；实际删除 **{receipt['deleted_files']:,} 个文件**，逻辑大小 **{receipt['deleted_logical_bytes']:,} 字节（{receipt['deleted_logical_bytes'] / 2**30:.2f} GiB）**。
- 文件校验后发生变化或删除失败而保留：{receipt['retained_after_verification']} 个；此前排除的 26 个文件仍存在且大小未变。
- D: 可用空间：{receipt['free_bytes_before']:,} → {receipt['free_bytes_after']:,} 字节；清理期间实测净增加 **{audit['free_space_delta_bytes'] / 2**30:.2f} GiB**。磁盘净变化可能包含其他进程写入以及文件系统分配开销，与删除逻辑字节数不完全相等。
- 仅删除已核对两端 SHA-256 且本地大小/修改时间未变化的任务副本；没有递归删除目录，代码、报告和其他文件保留。
- 清理回执：`{receipt_path.name}`；完整逐文件日志保留在本地操作目录，SHA-256 为 `{sha.hexdigest()}`。Git 保存回执、校验摘要和推送前置证据，未纳入大型数据或完整逐文件日志。

本次完成的是资产/环境准备、GitHub 报告提交和本地副本清理。真实 8B GPU 自检及 S1/S2 的通过状态仍须由独立实验日志确认，未运行 S4。
'''
report.write_text(text, encoding='utf8')
print(json.dumps(audit, ensure_ascii=False, indent=2))
