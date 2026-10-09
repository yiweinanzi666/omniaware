"""CPU continuation: verify live ownership, terminal evidence and failure ordering."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from . import continue_dense_cpu as C
from . import build_dense_index as B


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source, self.output = self.folder/'source', self.folder/'output'
        self.source.mkdir(); self.output.mkdir()
        C.save(self.source/'plan.json',dict(documents=2))
        self.owner = dict(pid=123,started='Tue Oct 6 12:00:00 2026',command='python encode fixed')
        self.bound = dict(directory=str(self.source),expected=2,completed_status='ENCODING_COMPLETE',
                         process=self.owner,plan_sha256=C.sha(self.source/'plan.json'))
        self.done = dict(status='ENCODING_COMPLETE',completed=2,expected=2,plan_sha256=self.bound['plan_sha256'])
        C.save(self.source/'completion.json',self.done)

    def test_paused_then_running_owner_is_waited_even_when_a_receipt_exists(self):
        states = [{123:dict(self.owner,state='T')},{123:dict(self.owner,state='S')},{}]
        with mock.patch.object(C,'processes',side_effect=states), mock.patch.object(C.time,'sleep') as sleep:
            C.wait_upstream('bge',self.bound,self.output)
        self.assertEqual(sleep.call_count,2)
        self.assertTrue((self.output/'bge_upstream_complete.json').exists())

    def test_terminal_owner_with_partial_output_stops(self):
        C.save(self.source/'completion.json',dict(self.done,completed=1,status='PARTIAL_CHECKPOINT'))
        with mock.patch.object(C,'processes',return_value={}):
            with self.assertRaisesRegex(ValueError,'without the complete'):
                C.wait_upstream('bge',self.bound,self.output)
        self.assertFalse((self.output/'bge_upstream_complete.json').exists())

    def test_reused_pid_does_not_hold_up_completed_original_work(self):
        other = dict(self.owner,started='Tue Oct 6 15:00:00 2026',command='unrelated',state='R')
        with mock.patch.object(C,'processes',return_value={123:other}), mock.patch.object(C.time,'sleep') as sleep:
            C.wait_upstream('bge',self.bound,self.output)
        sleep.assert_not_called()
        receipt = C.read(self.output/'bge_upstream_complete.json')
        self.assertEqual(receipt['current_pid_observation'],other)

    def test_another_encoder_lock_is_not_treated_as_finished(self):
        with (self.source/'worker.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(C,'processes',return_value={}), self.assertRaises(BlockingIOError):
                C.wait_upstream('bge',self.bound,self.output)

    def test_changed_bound_source_prevents_a_child_command(self):
        path = self.folder/'source.py'; path.write_text('before')
        plan = dict(files_sha256={str(path):C.sha(path)})
        path.write_text('after')
        with mock.patch.object(C.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'input changed'):
                C.run_stage('changed',['not-a-command'],self.output,plan)
        launch.assert_not_called()

    def test_real_failed_child_has_a_cost_and_error_receipt(self):
        with self.assertRaises(subprocess.CalledProcessError):
            C.run_stage('test_failure',[sys.executable,'-c','raise SystemExit(7)'],self.output,dict(files_sha256={}))
        receipt = C.read(next(self.output.glob('test_failure_*.json')))
        self.assertEqual((receipt['status'],receipt['returncode']),('failed',7))
        self.assertGreater(receipt['wall_seconds'],0)
        self.assertGreater(receipt['pid'],0)

    def test_failed_lane_allows_other_lane_but_never_fits_a_partial_corpus(self):
        lanes, final = C.stages()
        C.save(self.output/'plan.json',dict(lanes=lanes,final=final))
        finished = []
        def lane(kind,plan,output):
            if kind == 'bge':
                raise ValueError('incomplete text corpus')
            finished.append(kind)
        with mock.patch.object(C,'run_lane',side_effect=lane), mock.patch.object(C,'run_stage') as stage:
            with self.assertRaisesRegex(RuntimeError,'lane failure'):
                C.run(self.output)
        self.assertEqual(finished,['clip'])
        stage.assert_not_called()
        self.assertIn('bge',C.read(self.output/'failed.json')['errors'])

    def test_completed_clip_encoding_is_verified_without_rewriting_its_receipt(self):
        encoded = self.folder/'clip_wave1'; encoded.mkdir()
        C.save(encoded/'completion.json',self.done)
        before = (encoded/'completion.json').read_bytes()
        plan = dict(upstream=dict(clip={}),lanes=dict(clip=[
            ('clip_wave1',['python','encode','--output',str(encoded)]),
            ('clip_index',['python','index'])]))
        with mock.patch.object(C,'wait_upstream'), mock.patch.object(C,'run_stage') as stage, \
             mock.patch.object(B,'inspect_inputs',return_value=([],[],['image:30000000'],{})) as inspect:
            C.run_lane('clip',plan,self.output)
        self.assertEqual([c.args[0] for c in stage.call_args_list],['clip_index'])
        inspect.assert_called_once_with('clip',[encoded])
        self.assertEqual((encoded/'completion.json').read_bytes(),before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
