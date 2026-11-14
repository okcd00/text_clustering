import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_survey_results import export


def test_export_keeps_scenarios_separate_and_never_publishes_worker_stderr():
    static = {
        'protocol':'unified-static-v1', 'status':'partial',
        'metric_definitions': {},
        'methods':[{'id':'kmeans','title':'KMeans','track':'known-k'}],
        'cohorts':[{'id':'static','scenario':'known-k','n':2,'oracle_k':2}],
        'runs':[{'cohort_id':'static','method':'kmeans','seed':0,'status':'error',
                 'error':'sensitive stack trace'}],
    }
    routing = {
        'protocol':'routing-v2-lexical-heldout-labels', 'status':'executed',
        'inputs':{'public':{'support_n':3,'validation_n':2,'test_n':2}},
        'runs':[{'dataset':'public','method':'prototype','seed':0,'known_n':1,
                 'known_accepted_accuracy':None,'known_coverage':0.0,
                 'hidden_false_acceptance':1.0,'test_routing_seconds':0.01,
                 'per_sample_data':'must not publish'}],
    }
    result = export(static, routing)
    assert [c['scenario'] for c in result['cohorts']] == ['known-k','routing']
    assert result['runs'][0]['status'] == 'error'
    assert 'sensitive stack trace' not in str(result)
    assert 'per_sample_data' not in str(result)
    assert result['runs'][1]['metrics']['known_accepted_accuracy'] is None
    assert result['runs'][1]['metrics']['known_coverage'] == 0.0
    routing['runs'].append(dict(routing['runs'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        export(static, routing)
