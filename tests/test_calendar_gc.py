import gc
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_calendar_gc import DiagnosticStream
from benchmark_calendar_v2 import Stream


def test_gc_swap_parity_and_weakrefs():
    rng=np.random.default_rng(71)
    z=rng.normal(size=(4100, 8)).astype(np.float32)
    p={'threshold': .3, 'branching_factor': 50}
    original=Stream(z[:2048], 'rolling-2048', p, lambda: {})
    states=[]
    for force in (False, True):
        state=DiagnosticStream(z[:2048], 'rolling-2048', p, lambda: {})
        state.force_gc=force
        states.append(state)
    # Disable automatic collection only in this structural unit test to make
    # the pre-collection weakref assertion deterministic, never in the worker.
    was_enabled=gc.isenabled()
    gc.disable()
    try:
        for row in z[2048:]:
            original.update(row)
            for state in states: state.update(row)
    finally:
        if was_enabled: gc.enable()
    for state in states:
        assert state.arrivals == original.arrivals
        np.testing.assert_array_equal(state.model.subcluster_centers_,original.model.subcluster_centers_)
        np.testing.assert_array_equal(state.model.predict(z),original.model.predict(z))
        assert len(state.events)==1
        assert state.events[0]['retired_alive_after_swap'] > 0
    assert states[1].events[0]['retired_alive_after_gc_or_noop']==0
    assert states[1].events[0]['gc_collected_objects'] > 0
    gc.collect()
    assert all(ref() is None for s in states for ref in s.retired_refs)


def test_export_fails_on_prediction_or_public_daily_mismatch(tmp_path):
    import copy
    import json
    import pytest
    from summarize_calendar_gc import main, ROOT
    published=json.loads((ROOT/'results/calendar-paired-v2/results.json').read_text())
    control=copy.deepcopy(next(r for r in published['runs'] if r['method']=='rolling-2048'))
    control['private_prediction_digest']='synthetic-unit-test-digest'
    (tmp_path/'gc-uninstrumented.json').write_text(json.dumps(control))
    for arm,rep in [('automatic',1),('explicit-after-swap',1),('explicit-after-swap',2),('automatic',2)]:
        run=copy.deepcopy(control)
        run.update(diagnostic_arm=arm,diagnostic_script_sha256='test-only',gc_thresholds=[700,10,10])
        for event in run['rebuild_events']:
            event.update(all_retired_alive_after_gc_or_noop=0,explicit_gc_seconds=0)
        (tmp_path/f'gc-{arm}-{rep}.json').write_text(json.dumps(run))
    output=tmp_path/'aggregate.json'
    main(tmp_path,output)
    assert 'private_prediction_digest' not in output.read_text()
    p=tmp_path/'gc-automatic-1.json';run=json.loads(p.read_text())
    run['private_prediction_digest']='changed';p.write_text(json.dumps(run))
    with pytest.raises(AssertionError):main(tmp_path,output)
    control['windows'][0]['n']+=1
    (tmp_path/'gc-uninstrumented.json').write_text(json.dumps(control))
    with pytest.raises(AssertionError):main(tmp_path,output)
