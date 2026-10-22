# Working Context

Text-only EmbeddingGemma 2 conversion experiment. Python 3.11 dependencies are
locked in `uv.lock` and installed in `.venv`. Model-class imports, CLI behavior,
and smoke-script input guards have been checked. The official checkpoint is
downloaded and hash-verified. The user-run original-model smoke test passed on
2026-10-16. A saved CPU float32 baseline passed capture validation on
2026-10-17; Core ML conversion is untested.

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
- `scripts/original/capture_baseline_f32.py`: verified original-model baseline capture.
- `models/embeddinggemma-2/`: official local checkpoint, ignored by Git.
- `models/receipts/embeddinggemma-2.json`: pinned revision and local file hashes.
- `artifacts/baseline-f32-512/initial/`: saved baseline and provenance snapshots, ignored.

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

For the same query, document texts, and prompt strings, the sibling project's
saved original-model baseline (`../embeddinggemma-coreml/artifacts/baseline-f32/metadata.json`)
records these cosine scores alongside the user-reported EmbeddingGemma 2 run:

| Document | EmbeddingGemma 1 baseline | EmbeddingGemma 2 smoke run |
| --- | --- | --- |
| Mars | 0.645413 | 0.902646 |
| Venus | 0.259444 | 0.686095 |
| Saturn | 0.232551 | 0.667635 |

The ranking is unchanged. The Mars-to-Venus margin is approximately 0.385969
for the old model and 0.216551 for the new one. Absolute cosine scales differ
between models; higher scores do not establish better retrieval quality.

With the official checkpoint available locally:

```sh
uv sync --locked
.venv/bin/python scripts/original/smoke_test_f32.py
```

The saved baseline contains 13 text fixtures: the Mars example, paraphrases,
unrelated and short text, French/Arabic/Japanese examples, and exact 511/512-token
documents. Each has unpadded and right-padded `(1, 512)` inputs and a normalized
float32 `(1, 768)` embedding. The selected variant limit is 512 prompted tokens;
this does not assess the model's full context window.

Capture verified all 14 model file hashes, checked the official pipeline against
public `encode()` (maximum absolute difference `0`), and checked padding
invariance (maximum absolute difference approximately `2.55e-7`). Six rejection
checks passed: empty, whitespace-only, and exact 513-token inputs for each task
prompt. Mars retrieval ranking was unchanged. Saved array checks, provenance
hashes, and refusal to overwrite existing evidence also passed.

`fixtures.npz` stores the exact tensor inputs and embeddings; `metadata.json`
records prompts, texts, lengths, feature metadata, environment, tolerances, and
hash links. Byte-for-byte snapshots of the receipt, capture script, and lockfile
are retained alongside them. This is original-model evidence; no Core ML package
has been exported or assessed.

To capture another run, choose a fresh directory:

```sh
.venv/bin/python scripts/original/capture_baseline_f32.py \
  --output artifacts/baseline-f32-512/another-run
```
