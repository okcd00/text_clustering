"""Project evaluated cohorts into a stable, aggregate-only public chart payload."""
import argparse
import json
import math
from pathlib import Path

STATIC_FIELDS = {
    'ari_all_noise_as_one_cluster', 'nmi_all_noise_as_one_cluster', 'coverage',
    'ari_covered', 'clusters_excluding_noise',
}
ROUTING_DEFINITIONS = {
    'known_coverage': 'Known test samples accepted / all known test samples.',
    'known_accepted_accuracy': 'Correctly routed known samples / accepted known samples; null when none accepted.',
    'known_correct_acceptance': 'Correctly routed and accepted known samples / all known test samples.',
    'hidden_false_acceptance': 'Accepted hidden test samples / all hidden test samples (lower is better).',
    'test_routing_seconds': 'Batch test routing wall time, including building scores from support; excludes representation preparation and threshold calibration.',
}
ROUTING_FIELDS = set(ROUTING_DEFINITIONS) | {
    'known_n', 'hidden_n', 'known_accepted_n', 'hidden_accepted_n', 'known_correct_accepted_n',
}


def numbers(record, allowed):
    return {
        name: record[name] for name in sorted(allowed & record.keys())
        if record[name] is None or (
            type(record[name]) in (int, float) and math.isfinite(record[name])
        )
    }


def export(static, routing):
    if static.get('protocol') != 'unified-static-v1' or routing.get('protocol') != 'routing-v2-lexical-heldout-labels':
        raise ValueError('Unexpected benchmark protocol; update the exporter explicitly.')
    if static.get('status') not in {'executed', 'partial'} or routing.get('status') != 'executed':
        raise ValueError('Only attempted final runs can be exported.')
    info = {'id', 'dataset', 'representation', 'dimensions', 'n', 'scenario', 'information_budget', 'oracle_k'}
    cohorts = [{k: c[k] for k in sorted(info & c.keys())} for c in static['cohorts']]
    methods = [{k: m[k] for k in ('id','title','track')} for m in static['methods']]
    methods += [
        {'id':'prototype', 'title':'原型余弦路由', 'track':'routing'},
        {'id':'class-max-1nn', 'title':'类别最大 1-NN', 'track':'routing'},
    ]
    runs = []
    for r in static['runs']:
        row = {k: r[k] for k in ('cohort_id','method','seed','status')}
        if r['status'] == 'ok':
            row.update(numbers(r, {'fit_predict_seconds','process_peak_rss_bytes'}))
            row['metrics'] = numbers(r['metrics'], STATIC_FIELDS)
        else:
            row['error'] = '本次配置未完成，未记录得分。'
        runs.append(row)
    for dataset, spec in routing['inputs'].items():
        cohort_id = f'{dataset}__train-lexical__routing'
        cohorts.append({
            'id':cohort_id, 'dataset':dataset, 'representation':'仅支持集拟合的 TF-IDF + SVD64',
            'n':spec['test_n'], 'dimensions':64, 'scenario':'routing',
            'information_budget':(
                f"有标签支持集 {spec['support_n']} 条；仅用验证集 {spec['validation_n']} 条校准拒识；"
                '验证未知类与测试未知类互斥。'
            ),
        })
        for r in routing['runs']:
            if r['dataset'] != dataset:
                continue
            runs.append({
                'cohort_id':cohort_id, 'method':r['method'], 'seed':r['seed'], 'status':'ok',
                **numbers(r, {'test_routing_seconds'}),
                'metrics':numbers(r, ROUTING_FIELDS - {'test_routing_seconds'}),
            })
    keys = [(r['cohort_id'], r['method'], r['seed']) for r in runs]
    cohort_ids = {c['id'] for c in cohorts}
    method_tracks = {m['id']:m['track'] for m in methods}
    tracks = {c['id']:c['scenario'] for c in cohorts}
    if len(keys) != len(set(keys)) or {k[0] for k in keys} != cohort_ids or any(
        method_tracks.get(method) != tracks[cohort] for cohort,method,_ in keys
    ):
        raise ValueError('Invalid or duplicate result rows, or methods mixed across scenarios.')
    return {
        'schema_version': 1, 'protocol': 'survey-comparison-v1', 'status': 'executed',
        'protocols': {'static':static['protocol'], 'routing':routing['protocol']},
        'cohorts':cohorts, 'methods':methods, 'runs':runs,
        'metric_definitions': {**static['metric_definitions'], **ROUTING_DEFINITIONS},
        'deferred': [
            '层级归属与持续演化需要独立的树结构与时间窗协议；本轮不以旧纪录补分。'
        ],
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('static', 'routing', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    data = export(json.loads(args.static.read_text()), json.loads(args.routing.read_text()))
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
