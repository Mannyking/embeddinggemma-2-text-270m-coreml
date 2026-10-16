# Working Context

Text-only EmbeddingGemma 2 conversion experiment. Python 3.11 dependencies are
locked in `uv.lock` and installed in `.venv`. Model-class imports, CLI behavior,
and smoke-script input guards have been checked. The official checkpoint is
downloaded and hash-verified. The user-run original-model smoke test passed on
2026-10-16; Core ML conversion is untested.

The official processor requires Pillow and Torchvision at initialization even
for text-only use. Both are explicit locked dependencies. Offline processor
loading and text tokenization passed after adding them.

The sibling `embeddinggemma-coreml` project is a workflow reference. Its
baselines, packages, and results do not establish evidence for this model.

## Layout

- `README.md`: project overview.
- `CONTEXT.md`: working context and current status.
- `AGENTS.md`: working rules.
- `.gitignore`: excludes local environments, weights, and generated artifacts.
- `pyproject.toml`, `uv.lock`, `.python-version`: original-model environment.
- `scripts/original/smoke_test_f32.py`: offline CPU float32 text smoke test.
- `models/embeddinggemma-2/`: official local checkpoint, ignored by Git.
- `models/receipts/embeddinggemma-2.json`: pinned revision and local file hashes.

Add directories as implementation requires them.

Model source: `google/embeddinggemma-2` at revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`. All 14 downloaded files
(about 1.53 GB) match upstream file metadata: LFS SHA-256 for weights and Git
blob SHA-1 for other files. The generated receipt records SHA-256 for each file;
upstream metadata is not independently authenticated. The unchanged checkpoint
includes multimodal weights; the smoke script disables unused encoders at load.

The smoke test expects a local official checkpoint at `models/embeddinggemma-2/`
(or `--model-path`). It loads the model with vision/audio disabled and eager
attention, uses the checkpoint's `SearchQuery` and `Document` prompts, rejects
empty or over-limit prompted text without truncation, and checks 768-dimensional
finite unit vectors and Mars retrieval ranking. It prints results only; it
creates no baseline or Core ML evidence.

The reported smoke run used CPU float32, eager attention, and right padding.
Prompted token lengths were 18 for the query and 16/14/13 for the documents.
All four embeddings passed shape, finiteness, and unit-norm checks. Retrieval
ranked Mars (`0.902646`), Venus (`0.686095`), then Saturn (`0.667635`). These
terminal-only results are an original-model sanity check, not a saved baseline
or Core ML assessment.

With the official checkpoint available locally:

```sh
uv sync --locked
.venv/bin/python scripts/original/smoke_test_f32.py
```
