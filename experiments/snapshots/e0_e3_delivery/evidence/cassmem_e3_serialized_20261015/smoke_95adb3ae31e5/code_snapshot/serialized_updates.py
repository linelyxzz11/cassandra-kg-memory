"""Opt-in, single-process/single-owner serialized updates.

One coordinator must own ALL writes to a backend/layout/index namespace.
This is not a distributed lock, durable journal, or atomic reader snapshot.
After restart, explicitly reconcile and adopt complete state before accepting
updates. Never infer readiness from the backend raw version alone.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np


class VersionConflict(ValueError):
    """Same version reused with different content."""


@dataclass
class _State:
    lock: object = field(default_factory=threading.Lock)
    admitted: int = -1
    completed: int = -1
    fingerprint: str | None = None
    relations: set = field(default_factory=set)
    initialized: bool = False


class SerializedUpdateService:
    def __init__(self, backend, cell, index):
        self.backend, self.cell, self.index = backend, cell, index
        self._registry_lock = threading.Lock()
        self._states = {}

    def _state(self, record):
        key = (record.scope_id, record.memory_id)
        with self._registry_lock:
            return self._states.setdefault(key, _State())

    @staticmethod
    def _payload(record, vector):
        if not isinstance(record.version, int) or isinstance(record.version, bool) or record.version < 0:
            raise ValueError('version must be a nonnegative integer')
        # Own a copy so caller mutation cannot alter a queued update.
        vector = np.array(vector, dtype=np.float32, copy=True)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError('invalid embedding')
        if hashlib.sha256(vector.tobytes()).hexdigest() != record.embedding_sha256:
            raise ValueError('embedding does not match record fingerprint')
        fingerprint = hashlib.sha256(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return vector, fingerprint

    def verify(self, record, vector, relations=()):
        """Content check, not an atomic publication barrier; caller serializes writes."""
        b, i, s, mid = self.backend, self.index, record.scope_id, record.memory_id
        graph = b.fetch_graph(self.cell, s, mid)
        if graph != record.graph_projection() or b.candidate_projection(self.cell, s, mid) != record.rawerk:
            return False
        wanted = {e['relation'] for e in record.edges}
        for relation in set(relations) | wanted:
            if (mid in b.ids_by_relation(self.cell, s, relation)) != (relation in wanted):
                return False
        if mid not in b.ids(self.cell, s):
            return False
        with i._scope_locks[s]:
            sp, de = i._sparse[s], i._dense[s]
            if i._sparse_versions.get((s, mid)) != record.version or i._dense_versions.get((s, mid)) != record.version:
                return False
            if mid not in sp.memory_ids or mid not in de.memory_ids:
                return False
            if sp.documents[sp.memory_ids.index(mid)] != record.rawerk:
                return False
            expected = vector / max(float(np.linalg.norm(vector)), 1e-12)
            return bool(np.allclose(de.normalized_embeddings[de.memory_ids.index(mid)], expected, rtol=1e-6, atol=1e-7))

    def adopt(self, record, vector, *, known_relations=()):
        """Bootstrap an already loaded, fully reconciled record, once per owner."""
        vector, fingerprint = self._payload(record, vector)
        state = self._state(record)
        with state.lock:
            if state.initialized:
                raise RuntimeError('record already adopted')
            relations = set(known_relations) | {e['relation'] for e in record.edges}
            if not self.verify(record, vector, relations):
                raise ValueError('bootstrap state is not fully ready')
            state.admitted = state.completed = record.version
            state.fingerprint = fingerprint
            state.relations = relations
            state.initialized = True

    def apply(self, record, vector, *, hook=None):
        """Apply revision, or return stale/duplicate without backend mutation.

        Hooks are diagnostic only. Hook exceptions propagate, release the lock,
        and retain admission watermark. Same-payload retries can repair a partial
        update; conflicting same-version payloads never enter the write pipeline.
        """
        vector, fingerprint = self._payload(record, vector)
        state = self._state(record)
        emit = hook or (lambda event, stage, ns: None)
        began = time.perf_counter_ns()
        emit('waiting', 'lock', began)
        with state.lock:
            acquired = time.perf_counter_ns()
            emit('acquired', 'lock', acquired)
            try:
                if not state.initialized:
                    raise RuntimeError('adopt a reconciled baseline before updating')
                with self.index._scope_locks[record.scope_id]:
                    dimension = self.index._dense[record.scope_id].normalized_embeddings.shape[1]
                if vector.shape != (dimension,):
                    raise ValueError('embedding dimension differs from active index')
                if record.version < state.admitted:
                    return {'status': 'stale', 'version': record.version, 'admitted': state.admitted}
                if record.version == state.admitted:
                    if fingerprint != state.fingerprint:
                        raise VersionConflict('same version has a different payload')
                    if state.completed == record.version:
                        return {'status': 'duplicate', 'version': record.version}
                # Advance before the first mutation; retain on partial failure.
                state.admitted, state.fingerprint = record.version, fingerprint
                state.relations.update(e['relation'] for e in record.edges)
                def canonical():
                    if self.backend.fetch_graph(self.cell, record.scope_id, record.memory_id) != record.graph_projection():
                        raise AssertionError('canonical content mismatch')
                    if self.backend.candidate_projection(self.cell, record.scope_id, record.memory_id) != record.rawerk:
                        raise AssertionError('projection content mismatch')
                def ready():
                    if not self.verify(record, vector, state.relations):
                        raise AssertionError('target readiness failed')
                operations = (
                    ('raw', lambda: self.backend.commit_raw(self.cell, record)),
                    ('structured', lambda: self.backend.write_structured(self.cell, record)),
                    ('canonical_check', canonical),
                    ('sparse', lambda: self.index.upsert_sparse(record.scope_id, record.memory_id, record.rawerk, record.version)),
                    ('dense', lambda: self.index.upsert_dense(record.scope_id, record.memory_id, vector, record.version)),
                    ('ready', ready),
                )
                for stage, operation in operations:
                    emit('before', stage, time.perf_counter_ns())
                    operation()
                    emit('after', stage, time.perf_counter_ns())
                state.completed = record.version
                return {'status': 'applied', 'version': record.version,
                        'lock_wait_ms': (acquired-began)/1e6,
                        'lock_hold_ms': (time.perf_counter_ns()-acquired)/1e6}
            finally:
                emit('releasing', 'lock', time.perf_counter_ns())
