"""Supported-parent hierarchy preflight. No encoder or algorithm execution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from prepare_datasets import prepare, read_rows, normalize, external

PROTOCOL = 'massive-hierarchy-supported-parent-v1'
SUPPORT_CAP = 20
QUERY_CAP_PER_LEAF = 8


def rank(items, namespace):
    return sorted(items, key=lambda r: hashlib.sha256(
        f"{PROTOCOL}|{namespace}|{r['id']}".encode()).digest())


def opaque(names, prefix):
    names = sorted(names, key=lambda s: hashlib.sha256(
        f'{PROTOCOL}|opaque|{prefix}|{s}'.encode()).digest())
    return {name: f'{prefix}{i:03d}' for i, name in enumerate(names)}


def construct(rows):
    """Returns method input, scorer-only records and aggregate audit.
    Never provide scorer or source maps to an evaluated method.
    """
    parents = defaultdict(set)
    for part in rows.values():
        for r in part:
            parents[r['label']].add(r['parent'])
    if any(len(v) != 1 or None in v for v in parents.values()):
        raise ValueError('Every leaf must have exactly one parent')
    parent_of = {label: next(iter(v)) for label, v in parents.items()}
    full_test = {normalize(r['text']) for r in rows['test']}
    full_validation = {normalize(r['text']) for r in rows['validation']}
    pools = {
        'support': [r for r in rows['train'] if normalize(r['text']) not in full_test | full_validation],
        'validation': [r for r in rows['validation'] if normalize(r['text']) not in full_test],
        'test': rows['test'],
    }
    # Within each source role remove every normalized duplicate group. Do not
    # choose a favorable representative when labels conflict.
    for role, items in pools.items():
        counts = Counter(normalize(r['text']) for r in items)
        pools[role] = [r for r in items if counts[normalize(r['text'])] == 1]
    counts = {role: Counter(r['label'] for r in items) for role, items in pools.items()}
    eligible = {leaf for leaf in parents if counts['support'][leaf] >= 2
                and counts['validation'][leaf] >= 1 and counts['test'][leaf] >= 1}
    by_parent = defaultdict(list)
    for leaf in eligible:
        by_parent[parent_of[leaf]].append(leaf)
    known, vh, th = set(), set(), set()
    # Data-schema-aware construction, NOT label-free split creation. No text
    # semantics or test performance are used to choose which leaf is hidden.
    for i, parent in enumerate(sorted(by_parent, key=lambda p: hashlib.sha256(
            f'{PROTOCOL}|parent|{p}'.encode()).digest())):
        leaves = sorted(by_parent[parent], key=lambda l: hashlib.sha256(
            f'{PROTOCOL}|leaf|{l}'.encode()).digest())
        if len(leaves) >= 3:
            vh.add(leaves[1]); th.add(leaves[2])
            known.update(leaves[:1] + leaves[3:])
        elif len(leaves) == 2:
            known.add(leaves[0]); (vh if i % 2 == 0 else th).add(leaves[1])
        else:
            known.update(leaves)
    if not known or not vh or not th:
        raise ValueError('Cannot create both disjoint hidden roles with supported parents')
    selected = {}
    for role, allowed in [('support', known), ('validation', known | vh), ('test', known | th)]:
        cap = SUPPORT_CAP if role == 'support' else QUERY_CAP_PER_LEAF
        selected[role] = [r for leaf in sorted(allowed)
                          for r in rank([r for r in pools[role] if r['label'] == leaf], role)[:cap]]
    for role in ('validation', 'test'):
        selected[role] = rank(selected[role], role + '-global-query-order')
    support_leaves = {r['label'] for r in selected['support']}
    support_parents = {parent_of[l] for l in support_leaves}
    if support_leaves != known or any(parent_of[l] not in support_parents for l in vh | th):
        raise ValueError('Unsupported known leaf or hidden parent after caps')
    for a, b in [('support','validation'),('support','test'),('validation','test')]:
        if {normalize(r['text']) for r in selected[a]} & {normalize(r['text']) for r in selected[b]}:
            raise ValueError('Cross-role text overlap')
    all_leaves = known | vh | th
    # Codes for all eligible leaves are scorer-private; method sees support
    # leaves only and cannot infer hidden leaves from original semantic names.
    leaves = opaque(all_leaves, 'L')
    pcs = opaque(support_parents, 'P')
    support = [{'text':r['text'], 'leaf_id':leaves[r['label']],
                'parent_id':pcs[r['parent']]} for r in selected['support']]
    nodes = [{'id':'R000', 'label':'R000', 'description':'', 'parent_id':None}]
    nodes += [{'id':pcs[p], 'label':pcs[p], 'description':'', 'parent_id':'R000'}
             for p in sorted(support_parents)]
    nodes += [{'id':leaves[l], 'label':leaves[l], 'description':'',
               'parent_id':pcs[parent_of[l]]} for l in sorted(known)]
    method = {'support':support, 'nodes':nodes,
              'validation_texts':[r['text'] for r in selected['validation']],
              'test_texts':[r['text'] for r in selected['test']]}
    scorer = {
        role:[{'leaf_id':leaves[r['label']], 'parent_id':pcs[r['parent']],
               'is_known':r['label'] in known} for r in selected[role]]
        for role in ['validation','test']
    }
    report = {
        'status':'preflight_only_no_model_or_algorithm_execution',
        'protocol':PROTOCOL,
        'source_leaf_count':len(parents),
        'eligible_leaf_count':len(eligible),
        'excluded_leaf_count':len(parents)-len(eligible),
        'known_leaf_count':len(known),
        'validation_hidden_leaf_count':len(vh),
        'test_hidden_leaf_count':len(th),
        'supported_parent_count':len(support_parents),
        'excluded_parent_count':len({next(iter(v)) for v in parents.values()})-len(support_parents),
        'cross_role_normalized_overlap':0,
        'unsupported_hidden_parents_after_caps':0,
        'split_fingerprint':hashlib.sha256(json.dumps(
            [sorted(known), sorted(vh), sorted(th)], ensure_ascii=False).encode()).hexdigest(),
        'limits':{'support_per_leaf':SUPPORT_CAP,'query_per_leaf':QUERY_CAP_PER_LEAF},
        'roles':{
            role:{'rows':len(items), 'leaves':len({r['label'] for r in items}),
                  'parents':len({r['parent'] for r in items}),
                  'known_rows':sum(r['label'] in known for r in items),
                  'hidden_rows':sum(r['label'] not in known for r in items)}
            for role, items in selected.items()
        },
        'supported_parent_hidden_counts':{
            pcs[p]:{'known_leaves':sum(parent_of[l]==p for l in known),
                    'validation_hidden_leaves':sum(parent_of[l]==p for l in vh),
                    'test_hidden_leaves':sum(parent_of[l]==p for l in th)}
            for p in sorted(support_parents)
        },
    }
    return method, scorer, report


def run(cache):
    audit = prepare('massive_zh_cn', Path(cache))
    _, _, result = construct(read_rows('massive_zh_cn', Path(cache)))
    result['source_revision'] = audit['source_revision']
    result['artifacts'] = audit['artifacts']
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    report=run(external(a.cache)); out=external(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Aggregate preflight only; method input and scorer labels stay in memory.')
