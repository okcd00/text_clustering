import sys,json,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_news_calendar import Stream,ari,snapshot_predict
P={'threshold':0.01,'branching_factor':50}
class CalendarTests(unittest.TestCase):
    def test_same_warmup_all_methods(self):
        x=np.eye(5,dtype=np.float32)
        streams=[Stream(x,m,P) for m in ['frozen','unbounded','cap-2048','cap-128','rolling-2048']]
        for s in streams:self.assertEqual(s.count(),5)
        for s in streams:np.testing.assert_array_equal(s.model.predict(x),streams[0].model.predict(x))
    def test_cap_stops_and_records_exact_first_skipped(self):
        s=Stream(np.eye(3,dtype=np.float32),'cap-3',P)
        s.update(np.ones(3));self.assertEqual(s.count(),3);self.assertEqual(s.first_freeze,1);self.assertEqual(s.skipped,1)
    def test_frozen_does_not_learn(self):
        s=Stream(np.zeros((2,3)),'frozen',P);s.update(np.ones(3));self.assertEqual(s.count(),1)
    def test_rolling_every_2048_counts_new_point_once(self):
        s=Stream(np.zeros((3,2)),'rolling-2048',P)
        for _ in range(2047):s.update(np.ones(2))
        self.assertEqual(s.rebuilds,0)
        s.update(np.ones(2));self.assertEqual(s.rebuilds,1)
        self.assertEqual(s.model.root_.subclusters_[0].n_samples_,2048)
        self.assertEqual(s.count(),1)
    def test_prediction_is_snapshot_not_future_fit(self):
        s=Stream(np.array([[0.,0.],[1.,1.]]),'unbounded',P)
        day=np.array([[100.,100.],[100.,100.]])
        before=snapshot_predict(s,day).copy()
        for x in day:s.update(x)
        self.assertTrue(np.all(before==1));self.assertTrue(np.all(s.model.predict(day)==2))
    def test_small_score(self):self.assertIsNone(ari([1],[0]))
if __name__=='__main__':unittest.main()
