import sys,json,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_calendar_v2 import Stream,score,memory
from prepare_news_calendar import split,stamp
ROOT=Path(__file__).resolve().parents[1]
class V2Tests(unittest.TestCase):
    def test_sampling_prefix(self):
        old=json.loads((ROOT/'experiments/calendar-preflight-v1/protocol.json').read_text())
        new=json.loads((ROOT/'experiments/calendar-paired-v2/protocol.json').read_text());new['id']=new['sampling_namespace']
        rows=[{'id':i,'time':stamp('2014-03-17')+i*3600000} for i in range(35*24)]
        a=split(rows,old);b=split(rows,new)
        self.assertEqual(a,b[:len(a)]);self.assertEqual(len(b),29)
    def test_empty_days_preserved(self):
        p=json.loads((ROOT/'experiments/calendar-paired-v2/protocol.json').read_text());p['id']=p['sampling_namespace']
        result=split([],p);self.assertEqual(len(result),29);self.assertTrue(all(not w[3] for w in result))
    def test_no_useless_buffer(self):
        s=Stream(np.zeros((2,2)),'frozen',{'threshold':.3,'branching_factor':50},lambda: {})
        self.assertIsNone(s.buffer)
    def test_overlap_probe_retains_old_tree(self):
        events=[];s=Stream(np.zeros((2,2)),'rolling-2048',{'threshold':.3,'branching_factor':50},lambda:events.append(1) or {})
        for _ in range(2048):s.update(np.ones(2))
        self.assertEqual(len(s.events),1);self.assertEqual(s.events[0]['arrival'],2048);self.assertEqual(len(events),2)
        self.assertEqual(s.model.root_.subclusters_[0].n_samples_,2048)
    def test_memory_counters(self):
        m=memory();self.assertEqual(set(m),{'VmRSS','RssAnon','RssFile','RssShmem'});self.assertGreater(m['VmRSS'],0)
    def test_empty_score(self):self.assertIsNone(score([],[]))
if __name__=='__main__':unittest.main()

class AggregateTests(unittest.TestCase):
    def test_repeat_guard_ignores_resource_only(self):
        from summarize_calendar_v2 import invariant
        a={'scores':{'story_ari':.3},'prediction_seconds':1,'after_update_memory':{'VmRSS':20},'private_prediction_digest':'same'}
        b={'scores':{'story_ari':.3},'prediction_seconds':2,'after_update_memory':{'VmRSS':30},'private_prediction_digest':'same'}
        self.assertEqual(invariant(a),invariant(b))
        b['scores']['story_ari']=.4;self.assertNotEqual(invariant(a),invariant(b))
    def test_worker_does_not_load_source_or_encoder(self):
        import ast
        s=ast.parse((ROOT/'scripts/benchmark_calendar_v2.py').read_text())
        imports=[]
        for n in ast.walk(s):
            if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
            if isinstance(n,ast.ImportFrom):imports.append(n.module)
        self.assertFalse(any(x.startswith(('torch','transformers','prepare_news_calendar','prepare_calendar_v2','benchmark_news_calendar')) for x in imports))
