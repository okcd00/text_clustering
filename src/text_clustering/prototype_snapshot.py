"""Non-executable snapshots of fitted prototype routing state.

Identities are caller-provided provenance, not authentication. Only load files
from trusted storage; JSON validation is not a signature or a resource sandbox.
"""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FIELDS = {'leaves', 'parents', 'leaf_parents', 'centers', 'owners', 'parent_centers'}
CONFIG = {'prototypes', 'seed', 'hierarchical', 'threshold', 'min_margin'}


def _identity(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('snapshot identities must be nonempty strings')
    return value


def _array(record):
    if not isinstance(record, dict) or set(record) != {'dtype', 'data'}:
        raise ValueError('invalid snapshot array record')
    try:
        if not isinstance(record['dtype'], str):
            raise ValueError('snapshot dtype must be a string')
        dtype = np.dtype(record['dtype'])
        # Object/byte labels are supported by some fit paths but deliberately
        # excluded from this portable format. Never use pickle or lossy str().
        if dtype.kind not in 'Uifu' or dtype.itemsize > 16384:
            raise ValueError('unsupported snapshot dtype')
        def scalars(value):
            if isinstance(value, list):
                for item in value:
                    yield from scalars(item)
            else:
                yield value

        original = list(scalars(record['data']))
        allowed = (str,) if dtype.kind == 'U' else (int, float)
        if any(type(value) not in allowed for value in original):
            raise ValueError('snapshot data does not match dtype')
        raw = np.asarray(record['data'])
        if raw.dtype.kind not in 'Uifu' or (dtype.kind == 'U') != (raw.dtype.kind == 'U'):
            raise ValueError('snapshot data does not match dtype')
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.asarray(record['data'], dtype=dtype)
        # Python scalar comparison preserves int/float precision differences;
        # NumPy promotion can round both sides of an int-to-float comparison.
        converted = list(scalars(result.tolist()))
        if len(converted) != len(original) or any(a != b for a, b in zip(converted, original)):
            raise ValueError('lossy snapshot array conversion')
        if dtype.kind != 'U' and not np.isfinite(result).all():
            raise ValueError('nonfinite snapshot array')
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('invalid snapshot array') from exc


def _restore(payload, encoder_identity, support_identity):
    from .workflows.prototype_routing import FrozenPrototypeRouting, _support_ids
    from .algorithms.prototype_routing import PrototypeRouter

    if not isinstance(payload, dict) or set(payload) != {
            'schema_version', 'encoder_identity', 'support_identity', 'config', 'state'}:
        raise ValueError('invalid prototype snapshot envelope')
    if type(payload['schema_version']) is not int or payload['schema_version'] not in (1, 2):
        raise ValueError('unsupported prototype snapshot schema')
    for key, expected in [('encoder_identity', encoder_identity), ('support_identity', support_identity)]:
        if _identity(payload[key]) != _identity(expected):
            raise ValueError(f'{key} mismatch')
    config, state = payload['config'], payload['state']
    if not isinstance(config, dict) or set(config) != CONFIG:
        raise ValueError('invalid snapshot configuration')
    tool = FrozenPrototypeRouting(**config)
    if not isinstance(state, dict) or set(state) != FIELDS:
        raise ValueError('invalid snapshot state fields')
    arrays = {name: _array(state[name]) for name in FIELDS}
    leaves, parents = arrays['leaves'], arrays['parents']
    for name in ('leaves', 'parents', 'leaf_parents'):
        a = arrays[name]
        if a.ndim != 1:
            raise ValueError('invalid snapshot ID dimensions')
        _support_ids(a, len(a), name)
    if len(leaves) < 2 or len(parents) < 1:
        raise ValueError('insufficient snapshot classes')
    if not np.array_equal(leaves, np.unique(leaves)) or not np.array_equal(parents, np.unique(parents)):
        raise ValueError('snapshot classes must be unique and sorted')
    leaf_parents = arrays['leaf_parents']
    if (leaf_parents.dtype != parents.dtype or len(leaf_parents) != len(leaves)
            or not np.array_equal(np.unique(leaf_parents), parents)):
        raise ValueError('snapshot leaf-parent mapping mismatch')
    centers, parent_centers = arrays['centers'], arrays['parent_centers']
    if (centers.ndim != 2 or centers.shape[1] < 1 or centers.dtype != np.dtype('float32')
            or centers.shape[0] != len(leaves) * tool.prototypes
            or parent_centers.dtype != np.dtype('float32')
            or parent_centers.shape != (len(parents), centers.shape[1])):
        raise ValueError('snapshot center shape or dtype mismatch')
    zero_parents = np.all(parent_centers == 0, axis=1)
    if payload['schema_version'] == 2 and (tool.hierarchical or not zero_parents.any()):
        raise ValueError('schema 2 requires flat routing with undefined parent centers')
    for a in (centers, parent_centers):
        normalized = np.isclose(np.linalg.norm(a.astype(np.float64), axis=1),
                                1., rtol=0, atol=1e-5)
        # Only exact zero parent rows represent undefined flat proposals. Tiny
        # nonzero rows are corrupt state, not an alternative zero encoding.
        if a is parent_centers and payload['schema_version'] == 2 and not tool.hierarchical:
            normalized |= zero_parents
        if not normalized.all():
            raise ValueError('snapshot centers must be normalized')
    owners = arrays['owners']
    if owners.dtype.kind not in 'iu' or not np.array_equal(owners, np.repeat(np.arange(len(leaves)), tool.prototypes)):
        raise ValueError('snapshot prototype ownership mismatch')
    router = PrototypeRouter(tool.prototypes, tool.seed, tool.hierarchical)
    for name, a in arrays.items():
        setattr(router, name, a)
    tool._router = router
    return tool


def save(tool, path, *, encoder_identity, support_identity):
    router = tool._fitted_router()
    if any(getattr(tool, key) != getattr(router, key) for key in ('prototypes', 'seed', 'hierarchical')):
        raise ValueError('configuration changed since fit; refit before saving')
    schema_version = 2 if np.any(np.all(router.parent_centers == 0, axis=1)) else 1
    payload = dict(schema_version=schema_version, encoder_identity=_identity(encoder_identity),
                   support_identity=_identity(support_identity),
                   config={name: getattr(tool, name) for name in CONFIG},
                   state={name: {'dtype': getattr(router, name).dtype.str,
                                 'data': getattr(router, name).tolist()} for name in FIELDS})
    # Complete validation and encoding before touching an existing checkpoint.
    _restore(payload, encoder_identity, support_identity)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '.', delete=False) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    # Failed publication leaves the previous snapshot and a recoverable temp file.
    os.replace(temporary, path)


def load(path, *, expected_encoder_identity, expected_support_identity):
    _identity(expected_encoder_identity)
    _identity(expected_support_identity)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate snapshot JSON key')
            result[key] = value
        return result

    payload = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_keys)
    return _restore(payload, expected_encoder_identity, expected_support_identity)
