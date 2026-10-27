# Curated result exports — 2026-10-27

Start with the [release index](../../docs/releases/2026-10-27.md) for experiment
boundaries, denominators and known invalid/replacement runs.

All listed copies retain their original bytes. `MANIFEST.json` records SHA-256,
size and source location for result exports and the accompanying source snapshots.
Paths beginning with a workspace campaign name are provenance references, not
promises that the entire original workspace is included in this repository.

Small `complete.json` / `validate.json` files are historical run markers. The raw
observations needed to repeat every validation are not included in this compact
release. They must not be interpreted as a fresh experiment executed on checkout.

Verify the copied files, offline and without changing any artifacts:

```powershell
python experiments/maintenance/verify_release.py
```
