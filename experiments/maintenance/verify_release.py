"""Verify copied public artifacts without databases, models, or API requests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(manifest_path: Path, root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    seen = set()
    errors = []
    total = 0
    for row in manifest['files']:
        name = row['path']
        path = (root / name).resolve()
        if name in seen or not path.is_relative_to(root.resolve()):
            errors.append({'path': name, 'error': 'duplicate_or_unsafe_path'})
            continue
        seen.add(name)
        if not path.is_file():
            errors.append({'path': name, 'error': 'missing'})
            continue
        data = path.read_bytes()
        total += len(data)
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            errors.append({'path': name, 'error': 'size_or_hash_mismatch'})
    return {'passed': not errors, 'files': len(seen), 'bytes': total, 'errors': errors}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=root / 'results/release_20261027/MANIFEST.json')
    args = parser.parse_args()
    report = verify(args.manifest, root)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
