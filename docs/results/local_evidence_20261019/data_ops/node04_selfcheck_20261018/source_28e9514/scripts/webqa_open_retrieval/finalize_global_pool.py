"""Exact-text document deduplication and an auditable source-ID alias mapping."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib
import itertools
import json
import sqlite3

from omniaware import paths

HERE = paths.WEBQA_OPEN
ROOT = paths.ASSET_ROOT


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def main():
    manifest=json.loads((HERE/'GLOBAL_POOL_MANIFEST.json').read_text())
    con=sqlite3.connect('file:'+str(HERE/'global_fact_variants.sqlite')+'?mode=ro',uri=True)
    stats=Counter();output=HERE/'text_documents.jsonl'
    with output.open('w') as f:
        cursor=con.execute("SELECT body_sha,id,payload FROM facts WHERE modality='text' ORDER BY body_sha,id")
        for row_index,(body_sha,group) in enumerate(itertools.groupby(cursor,key=lambda r:r[0])):
            rows=list(group);payloads=[json.loads(r[2]) for r in rows]
            bodies={p['text'] for p in payloads};assert len(bodies)==1
            text=next(iter(bodies));assert hashlib.sha256(text.encode()).hexdigest()==body_sha
            ids=sorted({r[1] for r in rows})
            doc=dict(row_index=row_index,document_id='text-sha256:'+body_sha,modality='text',text=text,
                source_snippet_ids=ids,canonical_source_snippet_id=ids[0],
                titles=sorted({p['title'] for p in payloads}),urls=sorted({p['url'] for p in payloads}))
            f.write(canonical(doc)+'\n');stats['documents']+=1;stats['snippet_aliases']+=len(ids)
            if len(ids)>1:stats['documents_with_multiple_snippet_ids']+=1
            if not text.strip():stats['empty_documents']+=1
    assert stats['documents']==544489
    assert stats['snippet_aliases']==manifest['pool_and_asset_counts']['text_unique_ids']==787697
    assert stats['empty_documents']==0
    assert con.execute("SELECT COUNT(*) FROM (SELECT id,COUNT(DISTINCT body_sha) AS n FROM facts WHERE modality='text' GROUP BY id HAVING n>1)").fetchone()[0]==0
    image_range=con.execute("SELECT MIN(CAST(id AS INT)),MAX(CAST(id AS INT)),COUNT(DISTINCT id) FROM facts WHERE modality='image'").fetchone()
    assert image_range==(30000000,30389749,389750)
    con.close()
    manifest['canonical_text_documents']=dict(stats,rule='Exact Unicode text equality, no case/space/punctuation normalization; preserve every source snippet ID alias, title and URL.',
        document_id='text-sha256:<sha256(exact UTF8 text)>',row_order='ascending text SHA256',
        canonical_source_snippet_id='lexicographically smallest alias; fixed independently of queries/labels',
        upstream_evidence='BM25_full_retrieval_code_only.py line33 gives image offset544489; lines87-89 define fact2uniid/uniid2fact.',
        correspondence_limit='Exact unique-text count matches official code, but original fact2uniid.pkl row order is not locally verified; this output has its own explicit row order.')
    manifest['official_image_id_range']=dict(min=image_range[0],max=image_range[1],unique=image_range[2],contiguous=True)
    manifest['files']['text_documents.jsonl']=dict(path=str(output),sha256=sha(output),bytes=output.stat().st_size)
    manifest['official_scope_evidence']['upstream_manifest_sha256']=sha(HERE/'source/UPSTREAM.json')
    manifest['official_scope_evidence']['public_documents_total']=544489+389750
    manifest['official_scope_evidence']['paper_scope_difference']='Official public global pool contains 544489 unique exact text documents plus389750 image IDs =934239 documents; V3.8 p6 approximately390k text-and-image pool is not this verified combined count.'
    manifest['source_license']=dict(path=str(HERE/'source/license.txt'),sha256=sha(HERE/'source/license.txt'))
    manifest['finalizer']=dict(path=str(Path(__file__)),sha256=sha(__file__),completed_utc=datetime.now(timezone.utc).isoformat())
    # Validate all corpus outputs and the bound primary source payload after finalization.
    checks={name:sha(item['path'])==item['sha256'] for name,item in manifest['files'].items()}
    assert all(checks.values())
    trainval=manifest['projected_sources'][0]
    assert sha(trainval['path'])==trainval['payload_sha256']
    counts={}
    forbidden={'Q','A','question','answer','answers','references','qid','positive','negative','gold','txt_posFacts','txt_negFacts','img_posFacts','img_negFacts'}
    for name in ['text_corpus.jsonl','image_corpus.jsonl','text_documents.jsonl','id_conflicts.jsonl','missing_image_assets.jsonl','reusable_image_assets.jsonl']:
        count=0
        with (HERE/name).open() as f:
            for line in f:
                row=json.loads(line);assert not forbidden & row.keys(),(name,forbidden&row.keys());count+=1
        counts[name]=count
    assert counts['text_corpus.jsonl']==787697 and counts['image_corpus.jsonl']==389750 and counts['text_documents.jsonl']==544489
    assert counts['missing_image_assets.jsonl']+counts['reusable_image_assets.jsonl']==389750
    validation=dict(status='PASS',checked_utc=datetime.now(timezone.utc).isoformat(),output_sha256_matches=checks,row_counts=counts,
        forbidden_answer_question_role_keys_absent=True,all_source_snippet_ids_preserved=True,text_body_sha_collisions=0,
        source_trainval_payload_sha_verified=True,model_calls=0,images_decoded=0)
    (HERE/'GLOBAL_POOL_VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    manifest['validation']=dict(path=str(HERE/'GLOBAL_POOL_VALIDATION.json'),sha256=sha(HERE/'GLOBAL_POOL_VALIDATION.json'))
    (HERE/'GLOBAL_POOL_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report=HERE/'GLOBAL_POOL_REPORT.md';text=report.read_text()
    text=text.replace('**已按官方ID合并全部train、val、test公开事实；这是全局事实库存与资源核查，尚未创建embedding或FAISS索引。**',
        '**已完成全部train、val、test公开事实的全局池：544,489份精确正文去重文本＋389,750个图像ID，共934,239个检索文档。** 全部787,697个原snippet ID通过alias映射保留；尚未创建embedding或FAISS索引。')
    before='## 冲突与不丢失策略'
    addition='''## 全局文本去重与官方规模对齐

`text_corpus.jsonl`是787,697条按原snippet ID保存的完整来源库存。供后续文本编码使用的是`text_documents.jsonl`：按正文精确相等去重为544,489份，不做小写、空白或标点归一化；123,709份正文带有多个snippet ID，所有alias、title、URL均保留。检索返回的document_id能够映射回所有官方snippet IDs，没有丢弃原引用身份。

已读取固定官方BM25 notebook的代码字段，跳过所有输出值。其line33明确图像offset为544489；lines87–89使用fact2uniid及其逆映射，支持按正文事实去重的全局文本语义。此处独立得到完全相同的唯一正文数。原fact2uniid.pkl的行顺序未在本地验证，因此本交付采用明确的SHA256排序和自有row_index，不冒称原FAISS行号相同。

图像ID连续覆盖30000000–30389749，共389,750。已有公共镜像manifest的indexed_image_ids=389740与该原始公开图像ID数相差10；本池依据原数据保留所有ID，不为适配镜像而删图。V3.8约390k text-and-image总池与这里934,239个文档的总数不同，仍需论文实际corpus manifest解释。

'''
    text=text.replace(before,addition+before)
    text += '\n最终校验：`GLOBAL_POOL_VALIDATION.json`验证7项输出SHA、所有JSONL行数、无答案/问题/正负角色键、全部snippet aliases覆盖、正文hash一致和图像ID完整连续范围。\n'
    text += f'\n`text_documents.jsonl`：{output.stat().st_size:,} bytes；SHA256 `{sha(output)}`。\n'
    report.write_text(text)
    print(json.dumps({'status':'GLOBAL_PUBLIC_POOL_AND_TEXT_ALIAS_MAP_VALIDATED','documents':stats['documents'],'source_snippet_ids':stats['snippet_aliases'],
        'image_ids':389750,'validation':validation['status'],'manifest_sha256':sha(HERE/'GLOBAL_POOL_MANIFEST.json'),
        'report_sha256':sha(report),'text_documents_sha256':sha(output)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
