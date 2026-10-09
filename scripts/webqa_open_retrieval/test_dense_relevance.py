"""ECDF boundaries, rank invariance, candidate sufficiency and artifact lifecycle."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import calibrate_dense_relevance as C


def candidates(kind,values):
    return [dict(source_id=f'{kind}:{i}',index_row=i,rank=i+1,cosine=v) for i,v in enumerate(values)]


class DenseRelevanceTests(unittest.TestCase):
    def test_right_continuous_ecdf_and_no_extrapolation(self):
        pool=[-.5,0.,0.,.5]
        for score,expected in [(-1.,0.),(-.5,.25),(0.,.75),(.1,.75),(.5,1.),(1.,1.)]:
            self.assertEqual(C.percentile(pool,score),expected)

    def test_independent_modality_rescaling_preserves_selection(self):
        pools=dict(bge=[-.8,-.2,.1,.7],clip=[-.7,-.1,.3,.5])
        rows=dict(bge=candidates('bge',[.8,.2,.1,0.,-.4]),clip=candidates('clip',[.6,.4,.2,-.2,-.5]))
        original=C.select('fixed-query',rows,pools)
        transformed=copy.deepcopy(rows);rescaled={}
        for kind,a,b in [('bge',.5,.1),('clip',.8,-.1)]:
            rescaled[kind]=[a*x+b for x in pools[kind]]
            for item in transformed[kind]:item['cosine']=a*item['cosine']+b
        changed=C.select('fixed-query',transformed,rescaled)
        self.assertEqual([(r['source_id'],r['r_j']) for r in original['selected_sources']],
                         [(r['source_id'],r['r_j']) for r in changed['selected_sources']])

    def test_five_per_modality_suffices_even_with_ecdf_plateaus(self):
        pools=dict(bge=[-.5,0.,0.,.5],clip=[-.3,.1,.1,.4])
        rows={kind:candidates(kind,[.9,.8,.7,.6,.55,.52,.51,.5,.49,.2,0.,-.2]) for kind in C.KINDS}
        for qid in ('q1','q2','q3'):
            all_sources=C.select(qid,rows,pools)['selected_sources']
            prefix=C.select(qid,{k:v[:5] for k,v in rows.items()},pools)['selected_sources']
            self.assertEqual(all_sources,prefix)
            reverse=C.select(qid,{k:list(reversed(v[:5])) for k,v in rows.items()},pools)['selected_sources']
            self.assertEqual(prefix,reverse)

    def test_completed_reentry_and_incomplete_output_preservation(self):
        def loaded(path,kind,query_directory,engineering):
            is_cal=Path(query_directory).name=='webqa_calibration500_queries_cpu'
            count,prefix=(500,'cal') if is_cal else (256,'target')
            rows=[dict(qid=f'{prefix}-{i}',encoder=kind,candidates=candidates(kind,[.9,.7,.5,.3,.1])) for i in range(count)]
            identity=dict(directory=str(path),plan_sha256=prefix,completion_sha256='done',rankings_sha256='raw',
                index_plan_sha256=kind+'plan',index_sha256=kind+'index',ids_sha256=kind+'ids',availability={},vectors=1024)
            return rows,identity
        with tempfile.TemporaryDirectory() as folder,patch.object(C,'load_retrieval',side_effect=loaded):
            base=Path(folder);cal=base/'calibration';out=base/'merged'
            C.fit('bge','clip',cal,True)
            C.merge(cal,'bge','clip',out,True)
            files=[p for p in base.rglob('*') if p.is_file() and p.name!='worker.lock']
            before={str(p):C.sha(p) for p in files}
            self.assertFalse(C.fit('bge','clip',cal,True)['new_fit'])
            self.assertFalse(C.merge(cal,'bge','clip',out,True)['new_merge'])
            self.assertEqual(before,{str(p):C.sha(p) for p in files})
            (out/'completion.json').unlink()
            (out/'selected.jsonl').write_text('interrupted output requiring inspection\n')
            with self.assertRaisesRegex(ValueError,'existing'):
                C.merge(cal,'bge','clip',out,True)
            self.assertEqual((out/'selected.jsonl').read_text(),'interrupted output requiring inspection\n')


if __name__=='__main__':
    unittest.main()
