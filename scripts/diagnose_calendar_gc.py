"""Matched rolling lifecycle probe; v2 sampling/learning and worker remain unchanged.

Only explicit full collection immediately after model swap differs between arms.
Weak references observe retired linked leaf nodes without retaining them.
"""
import argparse
import gc
import json
import time
import weakref
import benchmark_calendar_v2 as base
from threadpoolctl import threadpool_limits


def leaf_refs(model):
    refs = []
    node = model.dummy_leaf_
    while node is not None:
        refs.append(weakref.ref(node))
        node = node.next_leaf_
    return refs


def alive(refs):
    return sum(ref() is not None for ref in refs)


class DiagnosticStream(base.Stream):
    force_gc = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retired_refs = []

    def update(self, x):
        # Delegate every non-rebuild insertion to the exact published worker.
        if self.buffer is None or (self.arrivals + 1) % 2048:
            return super().update(x)
        self.arrivals += 1
        self.buffer.append(x.copy())
        refs = leaf_refs(self.model)
        self.retired_refs.extend(refs)
        event = {'arrival': self.arrivals, 'old_leaf_cf': self.count(),
                 'retired_leaf_nodes_including_dummy': len(refs),
                 'before_memory': self.probe(), 'gc_count_before': gc.get_count()}
        other = self.new()
        for v in self.buffer:
            other.partial_fit(v[None, :])
        event.update(new_leaf_cf=len(other.subcluster_centers_), overlap_memory=self.probe())
        self.model = other
        event.update(after_swap_memory=self.probe(), retired_alive_after_swap=alive(refs),
                     all_retired_alive_after_swap=alive(self.retired_refs))
        start = time.perf_counter()
        collected = gc.collect() if self.force_gc else None
        elapsed = time.perf_counter() - start
        event.update(explicit_gc_seconds=elapsed if self.force_gc else None,
                     gc_collected_objects=collected, after_gc_or_noop_memory=self.probe(),
                     retired_alive_after_gc_or_noop=alive(refs),
                     all_retired_alive_after_gc_or_noop=alive(self.retired_refs),
                     gc_count_after=gc.get_count())
        self.events.append(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['automatic', 'explicit-after-swap'], required=True)
    args = parser.parse_args()
    DiagnosticStream.force_gc = args.arm == 'explicit-after-swap'
    base.Stream = DiagnosticStream
    with threadpool_limits(limits=1):
        result = base.main(base.external(args.cache), 'rolling-2048')
    result['diagnostic_arm'] = args.arm
    result['diagnostic_script_sha256'] = base.sha(base.ROOT/'scripts/diagnose_calendar_gc.py')
    result['gc_thresholds'] = gc.get_threshold()
    base.external(args.output).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
