"""LEGACY (not runnable from this repository).  Build a global public WebQA fact inventory, without decoding Q/A
or keeping roles.

This is the first pool builder of the experiment host.  It imports prepare_eab_recovery_confirmation.py from the
asset root (not shipped) and hard-codes a macOS path, so it cannot be rerun here.  It is kept because the
544,489 texts / 389,750 images of docs/GLOBAL_POOL_REPORT.md come from it.  The pool used now is built by
scripts/setup/prepare_webqa_open.py; its pool_stats.json holds the counts to report in the paper.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import os
import sqlite3
import subprocess

from omniaware import paths

HERE = paths.WEBQA_OPEN
ROOT = paths.ASSET_ROOT
spec=importlib.util.spec_from_file_location('projection',ROOT/'prepare_eab_recovery_confirmation.py')
projection=importlib.util.module_from_spec(spec);spec.loader.exec_module(projection)
FACT_FIELDS={'txt_posFacts','txt_negFacts','txt_Facts','img_posFacts','img_negFacts','img_Facts'}


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):(HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


class StructuralStream:
    """Buffer only several question records; skipped values are never JSON-decoded."""
    def __init__(self,stream):
        self.stream=stream;self.buffer='';self.eof=False;self.digest=hashlib.sha256();self.bytes=0
        self.decoder=__import__('codecs').getincrementaldecoder('utf-8')()
    def more(self):
        block=self.stream.read(1024*1024)
        if block:
            self.digest.update(block);self.bytes+=len(block);self.buffer+=self.decoder.decode(block)
        else:self.buffer+=self.decoder.decode(b'',final=True);self.eof=True
    def need(self):
        self.buffer=self.buffer.lstrip()
        while not self.buffer and not self.eof:self.more();self.buffer=self.buffer.lstrip()
    def value(self):
        while True:
            try:end=projection.value_end(self.buffer,0);break
            except (IndexError,ValueError):
                if self.eof:raise
                self.more()
        raw=self.buffer[:end];self.buffer=self.buffer[end:];return raw
    def records(self):
        self.need();assert self.buffer[0]=='{';self.buffer=self.buffer[1:]
        while True:
            self.need()
            if self.buffer.startswith('}'):
                self.buffer=self.buffer[1:]
                while not self.eof:self.more()
                assert not self.buffer.strip();return
            key_raw=self.value();json.loads(key_raw)  # top-level ID only, not exported
            self.need();assert self.buffer[0]==':';self.buffer=self.buffer[1:];self.need()
            yield self.value()
            self.need()
            if self.buffer.startswith(','):self.buffer=self.buffer[1:]
            elif not self.buffer.startswith('}'):raise ValueError('Expected object separator')


def image_assets():
    roots=sorted(p for p in (ROOT/'cache').glob('webqa*') if p.is_dir())
    records=defaultdict(dict);manifests=[]
    def actual_path(value):
        path=Path(value)
        if not path.is_absolute():return None
        relocated=Path(str(path).replace('/Users/futaoran/Downloads/WWW2027/experiments/eab_first_token',str(ROOT)))
        for candidate in [relocated,path]:
            if candidate.is_file() and candidate.stat().st_size>0:return candidate
        return None
    def walk(value,origin):
        if isinstance(value,dict):
            if 'image_id' in value and isinstance(value.get('local_path'),str):
                path=actual_path(value['local_path'])
                if path:
                    image_id=str(value['image_id']);size=path.stat().st_size
                    records[image_id][str(path)]=dict(path=str(path),bytes=size,
                        recorded_sha256=value.get('sha256'),manifest=str(origin),
                        recorded_bytes_match=value.get('bytes') in (None,size),
                        official_tsv_bytes_verified=value.get('official_tsv_bytes_verified',False),
                        current_validation='exists and byte-size checked; image payload not rehashed or decoded in this audit')
            for child in value.values():
                if isinstance(child,(dict,list)):walk(child,origin)
        elif isinstance(value,list):
            for child in value:walk(child,origin)
    for directory in roots:
        for path in sorted(directory.glob('*manifest*.json')):
            document=json.loads(path.read_text());walk(document,path)
            manifests.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
        for base,dirs,files in os.walk(directory):
            for name in files:
                path=Path(base)/name
                if path.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'} or not path.stem.isdigit():continue
                if path.stat().st_size==0:continue
                image_id=str(int(path.stem))
                records[image_id].setdefault(str(path),dict(path=str(path),bytes=path.stat().st_size,
                    recorded_sha256=None,manifest=None,recorded_bytes_match=None,official_tsv_bytes_verified=False,
                    current_validation='numeric filename and nonempty file only; ID/content identity requires verification before use'))
    return records,manifests,[str(p) for p in roots]


def main():
    database=HERE/'global_fact_variants.sqlite'
    if database.exists():raise FileExistsError('Preserve existing completed/partial corpus; explicit resume review required: '+str(database))
    con=sqlite3.connect(database)
    con.execute('PRAGMA journal_mode=DELETE');con.execute('PRAGMA cache_size=-32000')
    con.execute('CREATE TABLE facts (modality TEXT,id TEXT,payload_sha TEXT,payload TEXT,body_sha TEXT,occurrences INTEGER,train INTEGER,val INTEGER,test INTEGER,PRIMARY KEY(modality,id,payload_sha)) WITHOUT ROWID')
    counts=Counter();field_schemas={};source_records=[]
    insert='INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(modality,id,payload_sha) DO UPDATE SET occurrences=occurrences+1,train=train+excluded.train,val=val+excluded.val,test=test+excluded.test'
    archive=ROOT/'data/WebQA_data_first_release.7z';trainval=ROOT/'data/WebQA_train_val.json'
    process=None
    for name in ['train_val','test']:
        if name=='train_val':stream=trainval.open('rb')
        else:
            process=subprocess.Popen(['bsdtar','-xOf',str(archive),'WebQA_test.json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            stream=process.stdout
        parser=StructuralStream(stream);query_count=0
        for raw in parser.records():
            fields=projection.project(raw,FACT_FIELDS|{'split'})
            split=fields.get('split','test' if name=='test' else None)
            if split not in {'train','val','test'}:raise ValueError(('Unexpected public split',split))
            if query_count==0:field_schemas[name]=sorted(fields)
            if not FACT_FIELDS&fields.keys():raise ValueError('No recognized public fact fields')
            for key,value in fields.items():
                if key=='split':continue
                modality='text' if key.startswith('txt_') else 'image'
                for item in value:
                    if modality=='text':
                        ident=item.get('snippet_id',item.get('fact_id'))
                        payload=dict(text=item['fact'],title=item.get('title',''),url=item.get('url',''))
                        body=payload['text']
                    else:
                        ident=item['image_id'];payload=dict(caption=item.get('caption',''),image_url=item.get('imgUrl',''))
                        body=payload['caption']
                    if ident is None or not str(ident):raise ValueError('Missing public fact ID')
                    if not all(isinstance(x,str) for x in payload.values()):raise TypeError(('Non-string public fact field',ident))
                    encoded=canonical(payload);digest=hashlib.sha256(encoded.encode()).hexdigest()
                    con.execute(insert,(modality,str(ident),digest,encoded,hashlib.sha256(body.encode()).hexdigest(),1,int(split=='train'),int(split=='val'),int(split=='test')))
                    counts[modality+'_occurrences']+=1;counts[split+'_'+modality+'_occurrences']+=1
            counts[split+'_queries']+=1;query_count+=1
            if query_count%1000==0:
                con.commit();print(json.dumps(dict(stage='public_fact_projection',source=name,queries=query_count,bytes_read=parser.bytes)),flush=True)
        stream.close()
        if process:
            errors=process.stderr.read().decode();exitcode=process.wait();assert exitcode==0,errors
        source_records.append(dict(name=name,path=str(trainval if name=='train_val' else archive),member=None if name=='train_val' else 'WebQA_test.json',
            payload_sha256=parser.digest.hexdigest(),payload_bytes=parser.bytes,queries=query_count))
        con.commit()
    assert counts['train_queries']==36766 and counts['val_queries']==4966 and counts['test_queries']==7540
    print('All three splits projected; inspecting actual local image files.',flush=True)
    assets,asset_manifests,asset_roots=image_assets()
    outputs={name:(HERE/name).open('w') for name in ['text_corpus.jsonl','image_corpus.jsonl','id_conflicts.jsonl','missing_image_assets.jsonl','reusable_image_assets.jsonl']}
    stats=Counter();conflict_examples=[]
    def emit(name,value):outputs[name].write(canonical(value)+'\n')
    cursor=con.execute('SELECT modality,id,payload,occurrences,train,val,test FROM facts ORDER BY modality,id,payload_sha')
    for (modality,ident),group in itertools.groupby(cursor,key=lambda r:(r[0],r[1])):
        variants=list(group);payloads=[json.loads(v[2]) for v in variants]
        provenance={split:sum(v[4+i] for v in variants) for i,split in enumerate(['train','val','test'])}
        record=dict(fact_id=modality+':'+ident,source_id=ident,modality=modality,occurrences=sum(v[3] for v in variants),split_occurrences=provenance)
        stats[modality+'_unique_ids']+=1
        if len(variants)>1:stats[modality+'_metadata_variant_ids']+=1
        conflict=None
        if modality=='text':
            texts=sorted({v['text'] for v in payloads});titles=sorted({v['title'] for v in payloads});urls=sorted({v['url'] for v in payloads})
            record.update(text=texts[0] if len(texts)==1 else None,text_variants=texts if len(texts)>1 else [],titles=titles,urls=urls,
                embedding_eligible=len(texts)==1 and bool(texts[0].strip()))
            if len(texts)>1:conflict=dict(fact_id=record['fact_id'],kind='same_text_id_different_text',text_variants=texts,titles=titles,urls=urls)
            if record['embedding_eligible']:stats['unambiguous_nonempty_text_ids']+=1
            else:stats['text_ids_needing_resolution']+=1
            emit('text_corpus.jsonl',record)
        else:
            captions=sorted({v['caption'] for v in payloads});urls=sorted({v['image_url'] for v in payloads})
            candidates=list(assets.get(ident,{}).values())
            verified=[x for x in candidates if x.get('manifest') and x.get('recorded_bytes_match') is True]
            status='manifest_backed_size_match' if verified else 'unverified_numeric_filename_only' if candidates else 'missing_local_file'
            record.update(image_id=ident,captions=captions,image_urls=urls,asset_status=status,local_candidates=candidates,
                pixels_ready_for_encoding=False,pixels_readiness_note='Before encoding, verify selected file SHA/ID and processor; this audit did not decode pixels.')
            if len(urls)>1 or len(captions)>1:conflict=dict(fact_id=record['fact_id'],kind='same_image_id_metadata_variants',captions=captions,image_urls=urls,
                note='Metadata variants retained; not automatically a pixel identity conflict, no URL selected silently.')
            stats['images_'+status]+=1
            if not candidates:emit('missing_image_assets.jsonl',dict(image_id=ident,fact_id=record['fact_id'],captions=captions,image_urls=urls,reason=status))
            else:emit('reusable_image_assets.jsonl',dict(image_id=ident,fact_id=record['fact_id'],asset_status=status,local_candidates=candidates))
            emit('image_corpus.jsonl',record)
        if conflict:
            stats[conflict['kind']]+=1;emit('id_conflicts.jsonl',conflict)
            if len(conflict_examples)<20:conflict_examples.append(dict(fact_id=conflict['fact_id'],kind=conflict['kind']))
    for f in outputs.values():f.close()
    con.close()
    manifest=dict(status='GLOBAL_PUBLIC_FACT_UNION_COMPLETE_DENSE_INDEX_NOT_BUILT',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Global union of public facts from every official train/val/test question; not per-question candidate retrieval. Original fact IDs retained; roles and question-to-fact links not exported.',
        source_archive=dict(path=str(archive),sha256=sha(archive),members=['WebQA_test.json','WebQA_train_val.json','license.txt']),
        projected_sources=source_records,projected_field_schemas=field_schemas,query_and_occurrence_counts=counts,pool_and_asset_counts=stats,
        source_field_access='Only split and fact arrays decoded. Q/A/Qcate/topic/Guid values skipped structurally. Positive and negative field names selected only to pool their contents; role membership and per-query associations discarded.',
        source_splits_used=['train','val','test'],answer_values_decoded=False,question_values_decoded=False,query_fact_links_exported=False,gold_roles_exported=False,
        conflict_policy='Retain every metadata variant by source ID. Conflicting text IDs have text=null and embedding_eligible=false; no first/last-write wins. Image URL/caption variants retained for identity review.',
        conflict_examples=conflict_examples,asset_scan=dict(roots=asset_roots,manifests=asset_manifests,
            validation='File existence and size vs saved manifest; no image payload decode or all-image rehash. Numeric-only filename matches explicitly unverified. Cache scan is scoped, not whole-machine absence.'),
        official_scope_evidence=dict(readme=str(ROOT/'data/official_sources/WebQA_Baseline/README.md'),readme_sha256=sha(ROOT/'data/official_sources/WebQA_Baseline/README.md'),
            statement='Official README: full-scale retrieval over 544k text sources and 390k image sources across entire dataset.',
            upstream_manifest=str(HERE/'source/UPSTREAM.json'),upstream_manifest_sha256=sha(HERE/'source/UPSTREAM.json'),
            upstream_operation='T_unknown_modality_bm25.py lines31-32 loads all uniid2fact and extends all imgid2caption.',
            paper_scope_difference='OmniAware V3.8 p6 says approximately390k text and image facts; exact paper corpus identity is not established by the official global-pool count.'),
        pretrained_model_calls=0,new_image_downloads=0,dense_embeddings_created=False,faiss_index_created=False,mps_used=False,
        implementation=dict(path=str(Path(__file__)),sha256=sha(__file__),projection_path=str(ROOT/'prepare_eab_recovery_confirmation.py'),projection_sha256=sha(ROOT/'prepare_eab_recovery_confirmation.py')),
        files={name:dict(path=str(HERE/name),sha256=sha(HERE/name),bytes=(HERE/name).stat().st_size) for name in list(outputs)+['global_fact_variants.sqlite']})
    save('GLOBAL_POOL_MANIFEST.json',manifest)
    lines=['# WebQA 全局公开事实池准备','',
        '**已按官方ID合并全部train、val、test公开事实；这是全局事实库存与资源核查，尚未创建embedding或FAISS索引。**','',
        f'源包含train {counts["train_queries"]}、val {counts["val_queries"]}、test {counts["test_queries"]}题。train/val从既有JSON流式读入，test直接从原7z成员流式读取，未落盘包含答案的test原文件。只解码split和事实数组，跳过Q/A等值，输出不含正负标记或query→facts关联。','',
        f'全局去重后：**{stats["text_unique_ids"]:,}个文本ID、{stats["image_unique_ids"]:,}个图像ID**。官方README给出的量级是544k文本＋390k图像，官方full-scale代码构造整个uniid2fact并追加全部imgid2caption；本目录没有把每题候选列表冒称全局检索。V3.8约390k text-and-image总池的写法与此官方规模不同，论文具体池仍需绑定。','',
        '## 冲突与不丢失策略','',
        f'文本ID带多套元数据：{stats["text_metadata_variant_ids"]:,}；真正同ID不同正文：{stats["same_text_id_different_text"]:,}。后者保留所有text_variants，text=null、embedding_eligible=false。图像同ID不同caption或URL：{stats["same_image_id_metadata_variants"]:,}，完整写入id_conflicts.jsonl；元数据变体不自动等同像素冲突，也没有静默挑一个URL。','',
        f'可明确编码的非空文本ID：{stats["unambiguous_nonempty_text_ids"]:,}；需先处理的文本ID：{stats["text_ids_needing_resolution"]:,}。所有官方ID仍保留，未按文本相似性跨ID合并。','',
        '## 现有图像资产','',
        f'有manifest且实际文件字节数匹配：{stats["images_manifest_backed_size_match"]:,}；只有数值文件名匹配而缺乏manifest身份：{stats["images_unverified_numeric_filename_only"]:,}；扫描范围内没有本地文件：{stats["images_missing_local_file"]:,}。','',
        'reusable_image_assets.jsonl逐ID列出候选实路径、字节数、来源manifest及已记录SHA；missing_image_assets.jsonl列出每个缺图ID、公开URL/caption。已存在不等于本轮验证了像素正确性：本轮仅核存在和字节数，没有解码或重哈希全部图片。正式编码前仍需核验选定文件SHA、ID映射、processor和具体图像版本；官方TSV原字节与公共镜像图片的证据边界逐条保留。','',
        '扫描限定于本项目cache/webqa*目录和相应manifest，不声称其他磁盘完全没有缺失ID。没有下载图像，也没有用caption替代实际CLIP图片编码。','',
        '## 可复用交付','']
    for name,item in manifest['files'].items():lines.append(f'- `{name}`：{item["bytes"]:,} bytes；SHA256 `{item["sha256"]}`。')
    lines += ['', 'GLOBAL_POOL_MANIFEST.json保留原archive及两个source payload哈希、官方说明/代码固定revision、扫描范围、逐文件哈希和完整计数。SQLite只保存无标签事实变体，便于逐ID复查；JSONL可直接作为后续embedding输入的来源清单。','',
        '旧队列未停止或重启；本轮0模型调用、0图片下载、未进入MPS。下一步的真实编码/检索参数需依据新统一协议固定；这里没有把依赖测试、部分旧embedding或文本可编码数当成完整dense检索就绪。']
    (HERE/'GLOBAL_POOL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=manifest['status'],counts=stats,manifest_sha256=sha(HERE/'GLOBAL_POOL_MANIFEST.json'),report_sha256=sha(HERE/'GLOBAL_POOL_REPORT.md')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
