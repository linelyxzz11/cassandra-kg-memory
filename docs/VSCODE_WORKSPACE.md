# VS Code workspace

Open `CassMem.code-workspace` from the repository root. The workspace enables
Markdown preview, adds `src/` to Python analysis, reads local configuration from the
ignored `.env`, and provides a task for the service-independent protocol tests.

The public artifact has five working areas:

- `src/cassmem/`: reusable system implementation.
- `experiments/`: paper experiment entry points.
- `data/`: canonical inputs and frozen retrieval artifacts.
- `results/`: citation-facing outputs.
- `docs/`: design, claims, registry, and reproduction notes.

Run `Python: Select Interpreter` once after creating `.venv`, then execute
`CassMem: protocol tests` from **Terminal > Run Task**.
