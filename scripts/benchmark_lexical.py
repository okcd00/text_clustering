"""Isolated lexical sanity baseline, not a semantic method leaderboard.
Pinned public test splits are clustered transductively, never used for tuning.
No text, sample identifiers, assignments or feature matrices are emitted.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import scipy
import sklearn
import joblib
import threadpoolctl
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize as l2
from threadpoolctl import threadpool_limits
from prepare_datasets import prepare, read_rows, normalize, external

SEEDS = [0, 1, 2]

def select_rows(rows, dedup):
    """Sensitivity: exclude ALL groups repeated in test or seen in train/dev.
    No label-based choice of representative and no moving rows across splits.
    """
    from collections import Counter
    test = rows['test']
    if not dedup: return test
    prior = {normalize(r['text']) for split, items in rows.items() if split != 'test' for r in items}
    counts = Counter(normalize(r['text']) for r in test)
    return [r for r in test if normalize(r['text']) not in prior and counts[normalize(r['text'])] == 1]

def scores(labels, pred):
    labels, pred = np.asarray(labels), np.asarray(pred)
    mask = pred != -1
    return {'ari_all_noise_as_one_cluster': float(adjusted_rand_score(labels, pred)),
            'nmi_all_noise_as_one_cluster': float(normalized_mutual_info_score(labels, pred)),
            'coverage': float(mask.mean()), 'clusters_excluding_noise': len(set(pred[mask])),
            'ari_covered': float(adjusted_rand_score(labels[mask], pred[mask])) if mask.sum() > 1 else None}

def run(cache):
    result = {'protocol': 'lexical-v1', 'seeds': SEEDS, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__, 'threadpoolctl': threadpoolctl.__version__}, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'threads': 1, 'runs': [], 'inputs': []}
    for dataset in ['banking77', 'massive_zh_cn']:
        report = prepare(dataset, Path(cache)/dataset)
        rows = read_rows(dataset, Path(cache)/dataset)
        for dedup in [False, True]:
            items = select_rows(rows, dedup)
            texts, labels = [r['text'] for r in items], [r['label'] for r in items]
            track = 'duplicate-excluded' if dedup else 'official-test'
            base = {'dataset': dataset, 'track': track, 'n': len(items), 'oracle_test_present_k': len(set(labels))}
            result['inputs'].append({**base, 'excluded_rows': len(rows['test'])-len(items), 'missing_official_test_labels': sorted({r['label'] for r in rows['test']}-set(labels)), 'source_revision': report['source_revision'], 'artifacts': report['artifacts']})
            start = time.perf_counter()
            X = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=2, max_features=10000, sublinear_tf=True, lowercase=True, dtype=np.float64).fit_transform(texts)
            tfidf_seconds = time.perf_counter()-start
            for seed in SEEDS:
                start=time.perf_counter()
                Z=l2(TruncatedSVD(n_components=64, n_iter=7, random_state=seed).fit_transform(X))
                representation_seconds=time.perf_counter()-start
                for method in ['one-cluster', 'kmeans-oracle-k', 'hdbscan-fixed']:
                    start=time.perf_counter()
                    if method=='one-cluster': pred=np.zeros(len(labels), dtype=int)
                    elif method=='kmeans-oracle-k': pred=KMeans(n_clusters=len(set(labels)), n_init=10, max_iter=300, random_state=seed).fit_predict(Z)
                    else: pred=HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean', cluster_selection_method='eom', n_jobs=1, copy=True).fit_predict(Z)
                    result['runs'].append({**base, 'seed': seed, 'method': method, **scores(labels,pred), 'tfidf_seconds_shared': tfidf_seconds, 'svd_seconds_shared': representation_seconds, 'cluster_seconds': time.perf_counter()-start})
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=external(args.output)
    with threadpool_limits(limits=1): result=run(external(args.cache))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f'{len(result["runs"])} aggregate runs saved; no per-example outputs')
