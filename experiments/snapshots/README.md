# Frozen experiment source snapshots

These are **provenance snapshots, not a portable application or automatic test suite**.
Files are copied byte-for-byte from completed local delivery packages or named
workspace campaigns. Shared code repeated in a snapshot is historical; active
implementation lives under `src/cassmem/`.

| Directory | Origin |
|---|---|
| `e0_e3_delivery/` | Python sources retained in the E0/E3 delivery dated 2026-10-18 |
| `access_path_delivery/` | Python sources retained in the access-path delivery dated 2026-10-20 |
| `workspace/` | Candidate-budget, scope-isolation and ordinary-concurrency campaign sources |

Original scripts may refer to `D:/memorytable`, external frozen fixtures, model
caches, or private run directories. Those paths and code bytes are retained so
historical provenance is not confused with a new port. Raw fixtures, database
state, credentials, private API responses, logs and datasets are **not** supplied
by these snapshots. Some sources in a delivery were already sanitized by that
delivery's author; hashes here refer to the exact delivered copy.

Do not recursively execute scripts or tests in this directory. Launchers may
start database workloads, modify experiment namespaces or issue paid API requests.
To reproduce a campaign, first review its configuration, restore legally available
inputs in an isolated workspace, confirm the original model/protocol and budget,
and validate any path adaptations. No such experiments run during repository sync.

For runnable local regression tests, use the commands in the repository README.
For copied-file hashes and original workspace-relative locations, see
[`MANIFEST.json`](../../results/release_20261027/MANIFEST.json).
