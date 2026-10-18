import pathlib,json,hashlib,datetime,shutil
L=pathlib.Path('D:/WWW2027/data_ops/node04_v2_20261017')
R=L/'review_7ccf676';O=R/'docs/results';O.mkdir(exist_ok=True)
def read(n):return json.loads((L/n).read_text(encoding='utf8'))
inv=read('preparation_inventory_1631.json');pub=read('publish_webqa_1631.json');tests=read('lf_checkout_pytest_1631.json');gs=read('resume_prepare_1608.json');old=read('V2_ASSET_CHECK_20261017_initial.json')
assert pub['ok'] and tests['ok'] and gs['ok'] and inv['pip_check']['returncode']==0
checker=(L/'check_assets_1640.log').read_text(encoding='utf8')
missing=[line for line in checker.splitlines() if line.startswith('MISSING')]
allowed=['InternVL3.5-8B-HF','OK-VQA BM25 candidates','OK-VQA DPR-wiki candidates','HybridQA corpora','HybridQA rankings','VideoRAG rankings','VideoRAG frames']
assert all(any(line.startswith('MISSING '+name) for name in allowed) for line in missing),missing
sources={
'qwen8b':('Qwen/Qwen3-VL-8B-Instruct','0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
'reranker':('BAAI/bge-reranker-v2-m3','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'),
'nli':('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli','b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'),
'bartscore':('BARTScore ParaBank official Drive + facebook/bart-large-cnn config','37f520fa929c961707657b28798b30c003dd100b (config)')}
evidence=['qwen8b_verified_manifest.json','reranker_verified_manifest.json','nli_verified_manifest.json','bartscore_verified_manifest.json','source_csv_verified_0657.json','resume_prepare_1608.json','publish_webqa_1631.json','lf_checkout_pytest_1631.json','preparation_inventory_1631.json','bartscore_cpu_asset_check_0750.json','check_assets_1640.json','check_assets_1640.log','pip_freeze_1640.txt','webqa_probe_1620.json','webqa_raw_check_1620_server.json','webqa_raw_check_1620_local.json']
folder=O/'20261017_node04_preparation_evidence';folder.mkdir()
receipts=[]
for n in evidence:
    p=L/n;assert p.exists(),n;shutil.copyfile(p,folder/n)
    receipts.append({'file':n,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
record={'created':datetime.datetime.now().astimezone().isoformat(),'scope':'Assets, data and Python environment preparation; not S1/S2 experimental results.','model_sources':sources,'model_assets':inv['model_manifests'],'packages':inv['packages'],'data_loaders':pub['loaders'],'webqa_image_fact_closure':pub['image_fact_closure'],'gs':gs['gs'],'pytest':{'source_commit':tests['commit'],'passed':546,'skipped':1,'subtests_passed':21,'failed':0,'errors':0},'experimental_gates':'Full code review and actual 8B selfcheck/S1/S2 are separate, not claimed complete here. S4 forbidden.','evidence':receipts,'local_cleanup':'Only task data/cache; code/reports/other data retained. File deletion requires matching local/server SHA256 and a separately logged deletion receipt.'}
record.update(v2_asset_preparation_accepted=True,global_asset_checker_exit_code=read('check_assets_1640.json')['exit_code'],excluded_missing_asset_lines=missing)
with (O/'20261017_node04_preparation_manifest.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2,ensure_ascii=False)
rows=[]
for key,(src,rev) in sources.items():
    d=inv['model_manifests'][key];rows.append(f"| {src} | 本轮补齐、发布并校验 | {d['files']} | {d['bytes']:,} | `{key}_verified_manifest.json` |")
for key,name in [('bge-large-en-v1.5','BAAI/bge-large-en-v1.5'),('clip-vit-large-patch14','openai/clip-vit-large-patch14')]:
    d=old['assets']['model_'+key];rows.append(f"| {name} | 复用已有资产；完整性检查通过 | {d['files']} | {d['bytes']:,} | 初次资产审计 + 本次结构检查 |")
lines=[
'# node04 模型、数据与环境准备清单（2026-10-17）','',
'本报告记录服务器资产与环境准备、验收和本地副本清理依据。它不是 S1/S2 实验结果；真实 8B 自检和阶段关卡不能由文件完整性或 CPU pytest 代替。S4 未执行。','',
'## 服务器位置','',
'- 主资产：`/home/user/Futaoran/omniaware/assets`。',
'- 本轮使用的资产视图：`/home/user/Futaoran/omniaware_v2_assets_20261017_resume`；固定 M2-RAG checkout，其他资产复用原位置。',
'- 修复后的 GS 缓存：`/home/user/Futaoran/omniaware_v2_cache_20261017_resume`。',
'- Python：`/home/user/Futaoran/omniaware_v2_python311/bin/python`。',
'- LF 检出目录：`/home/user/Futaoran/omniaware_v2_exec_lf`，提交 `7ccf676e9804b2dfdf571d5b94fe8c9aba843acb`。',
'- 全部日志和文件校验清单：`/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/`。','',
'## 下载、复用的模型','',
'下面大小来自验证清单中的实际文件字节数。BGE/CLIP 为目录审计，可能包括少量下载元数据；不能据此宣称这些资产全部在本轮重新下载。','',
'| 模型 | 本轮操作 | 文件数 | 字节数 | 证据 |','|---|---|---:|---:|---|',*rows,'',
'模型官方来源与固定版本：','',
*[f'- `{k}`：`{s}`，版本 `{v}`。' for k,(s,v) in sources.items()],
'- BGE：`d4aa6901d3a41ba39fb536a557fa166f842b0e09`；CLIP：`32bd64288804d66eefd0ccbe215aa642df71cc41`（仓库下载器固定版本）。',
'- BARTScore ParaBank 官方来源：`https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m`。权重 `bart_score.pth` 为 1,625,565,295 字节，SHA-256 `4c076ee386795a831e0021788148becf57136127d6512ca350c7973f36c0e573`；未修改的加载器在 CPU 严格加载成功，无缺失/多余参数。',
'- Qwen、reranker、NLI、BARTScore 清单逐文件记录 SHA-256。Qwen 的四个分片以及 BGE/CLIP/reranker safetensors 结构完整性检查通过。','',
'## 数据及修复','',
'- OK-VQA：5,046 道 val 问题、5,033 张去重 COCO 图片；官方问题和标注 ZIP 存在，图片已全部解码检查；本次 GS 加载器返回 5,046，`dropped={}`。',
'- 官方 GS CSV 两份已上传并校验：train 41,252,142 字节，SHA-256 `f6646ad30d05166cfd32c281566a0032656035761e7d33acaf3267bfbb9d4997`；full 60,445,862 字节，SHA-256 `4ab2352a2bff6413ac74f1a18ef7ffcc8a15b5f77df97a5b6afd8375448128ae`。',
'- GS-112K：旧 corpus 的索引前缀与官方 CSV 逐条一致，尾部另有 112,724 字节异常内容。保留旧文件，在新目录生成恰好 112,724 行的 canonical corpus，并验证旧 offsets 与每个候选文本一致后复用向量和候选。新 corpus SHA-256 `058fe2fc84d9aad7e1dc2b6a2f39aeafe8b587da5380e895c5e3ced1b491a8e7`；向量 230,858,752 字节，done=112724；5,046 条候选记录，每条 rank 1–20。',
'- WebQA：train/val 标注、test 标注、72,234,224,104 字节原始 TSV 与 lineidx 在服务器；开放池和 train/val 的 BGE、CLIP ranking 复用已有资产。',
f"- WebQA train/val 共检查 {pub['image_fact_closure']['checked_facts']:,} 个图片事实，缺失 0。各加载器计数和 dropped 详见 JSON 清单。",
'',
'### WebQA 原始图片的明确处理例外','',
'10 张原始图片在本地与服务器的字节 SHA-256 完全一致，但严格 PIL 解码失败，影响 18 个 train 和 4 个 val 的负例图片事实。它们不是传输损坏。官方 WebQA README（提交 `81cc377aaddb2cf9de98cf082a2795ce53ccc1f4`）建议启用 `ImageFile.LOAD_TRUNCATED_IMAGES=True`。用户在本会话明确批准采用该官方读取方式并记录处理方法。',
'',
'处理仅针对以下 10 张图片：`30278219, 30045608, 30366390, 30298493, 30014375, 30039288, 30233625, 30161353, 30172755, 30039112`。从原始 TSV 解码，转 RGB，按现有准备脚本的 JPEG quality=95 保存，关闭容错后重新加载转换产物，再发布到此前缺失的路径。原始 TSV、旧图片和已有索引均保留。容错读取不保证恢复缺失像素；原始/转换后 SHA-256、尺寸、路径和受影响题号见证据。',
'',
'已有开放检索向量和 ranking 未重建；最初索引过程会排除不可解码图片。这次补图用于补齐 distractor 事实，不应把它描述为重新构建了包含新增图片的开放检索索引。','',
'## Python 环境与测试','',
*[f'- `{k}=={v}`' for k,v in inv['packages'].items()],
'- Python 3.11.17；`pip check` 返回 0，所有声明依赖及官方外部仓库固定提交已校验。全部安装版本见 `pip_freeze_1640.txt`。',
'- `python -m pytest -p no:warnings`：546 passed、1 skipped、21 subtests passed，0 failed、0 errors，退出码 0。',
'- 旧运行中的 M2-RAG 溯源失败通过新资产视图修复；localhost 下载测试通过 NO_PROXY 修复；官方 WebQA 评分脚本校验失败来自 `core.autocrlf=true` 的 CRLF 检出。在新的 LF 工作目录中，官方脚本字节哈希匹配固定值。',
'- `check_assets` 的完整输出和退出码原样保存；v2 不需要的 HybridQA、VideoRAG、InternVL、BM25/DPR 等缺项按原任务约定不作为本轮资产阻塞。',
'',
'使用本轮环境（执行实验前仍须完成其独立代码审查和真实模型关卡）：','',
'```bash','cd /home/user/Futaoran/omniaware_v2_exec_lf','source /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh','python -m pip check','```','',
'基础提交的真实模型参考解码器有已知 CPU/CUDA 掩码设备问题。用户已在“按 GitHub 要求推进实验”会话批准独立修复，该会话在服务器提交了 `da416a2266a0f3f01e0c394674d0e7b0ad0e5722`。本报告的完整 pytest 使用基础提交 7ccf676；不把它冒称为修复分支的最终 GPU 自检。本轮未修改该会话的工作目录和运行。','',
'## 本地清理范围与证据','',
'用户明确选择“仅清理本任务的数据和缓存，保留代码、报告及其他文件”。候选计划包含 394,890 个文件、225,973,716,961 字节（约 210.5 GiB），这只是候选总量，不是已释放空间。执行前逐文件比较本地与服务器 SHA-256；缺失、大小/哈希不同或校验后发生变化的文件保留。实际删除数量、字节数、未删除项和磁盘可用空间变化记录在后续清理回执。禁止递归删除整个 D: 或 D:/WWW2027。','',
'本提交只包含报告、小型校验清单与环境版本，不包含权重、数据集、密码或 SSH 私钥。',
]
with (O/'20261017_node04_preparation.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(str(O/'20261017_node04_preparation.md'))
