# Working Context

Text-only EmbeddingGemma 2 conversion experiment. Python 3.11 dependencies are
locked in `uv.lock` and installed in `.venv`. Model-class imports, CLI behavior,
and smoke-script input guards have been checked. The official checkpoint is
downloaded and hash-verified. The user-run original-model smoke test passed on
2026-10-16. A saved CPU float32 baseline passed capture validation on
2026-10-17. The saved float32 512-token Core ML package has passed CPU-only
parity assessment on all 13 fixtures. The LiteRT-LM int4 text bundle has also
been compared on CPU; it does not meet float32 parity.

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
- `pyproject.toml`, `uv.lock`, `.python-version`: original-model and conversion environment.
- `scripts/original/smoke_test_f32.py`: offline CPU float32 text smoke test.
- `scripts/original/capture_baseline_f32.py`: verified original-model baseline capture.
- `models/embeddinggemma-2/`: official local checkpoint, ignored by Git.
- `models/receipts/embeddinggemma-2.json`: pinned revision and local file hashes.
- `artifacts/baseline-f32-512/initial/`: saved baseline and provenance snapshots, ignored.
- `scripts/coreml/`: float32 export, saved-package assessment, and shared evidence checks.
- `artifacts/coreml/f32-512/numpy1/`: exported package and CPU assessment, ignored.
- `scripts/litert/`: pinned LiteRT-LM download and descriptive baseline comparison.
- `models/embeddinggemma-2-text-270m-litertlm/`: CPU/GPU bundle and model card, ignored.
- `models/receipts/embeddinggemma-2-text-270m-litertlm.json`: bundle revision and hashes.
- `artifacts/litert/text-270m/assessments/initial-cpu/`: saved LiteRT-LM comparison, ignored.

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
are retained alongside them. This is original-model evidence; Core ML parity
requires the separate saved-package assessment below.

To capture another run, choose a fresh directory:

```sh
.venv/bin/python scripts/original/capture_baseline_f32.py \
  --output artifacts/baseline-f32-512/another-run
```

## Float32 Core ML

`artifacts/coreml/f32-512/numpy1/EmbeddingGemma2F32.mlpackage` accepts `(1, 512)`
int32 `input_ids` and `attention_mask`, and returns a normalized `(1, 768)`
float32 `embedding`. Task prompting and tokenization stay outside the package.
Callers must reject empty or over-512-token prompted text and use right padding.
The deployment target is iOS 17/macOS 14; device behavior has not been assessed.

Export rechecked the pinned receipt, every local model hash, saved baseline
hashes, and original-pipeline outputs. Wrapper and trace errors were zero on all
13 fixed-input fixtures. The wrapper preserves upstream bidirectional full/local
masks (including elided all-zero masks), media-placeholder-to-PAD behavior,
per-layer inputs, projection, pooling, and normalization.

The saved package is about 1.09 GB. Its CPU-only assessment passed `f32-parity`
in `assessments/cpu-native/`: maximum absolute error `2.68e-7` against padded
reference embeddings and `3.61e-7` against unpadded ones, maximum norm error
`9.21e-8`, and unchanged Mars retrieval ranking. The profile matches the first
project: elementwise `atol=1e-4`, `rtol=1e-3`, cosine at least `0.9999`, and norm
error at most `1e-4`. Reports link package, baseline, source receipt, and script
hashes; predictions and source/lock snapshots are retained. This establishes
recorded-fixture parity on macOS CPU only, not performance or accelerator placement.

Conversion uses Core ML Tools 9.0 and NumPy 1.26.4. NumPy 2.4 caused the
converter's scalar cast to fail. Torch 2.14.1 is outside Core ML Tools' tested
versions; the saved-package results above are the actual conversion evidence.
The wrapper expresses media-placeholder replacement with `where` to avoid an
unsupported `__or__` operation in the composite model.

Earlier failed export reports remain in `initial/`, `mask-elision/`, and
`text-backbone/` under `artifacts/coreml/f32-512/`. The accepted package's
`assessments/initial/` records a sandbox compilation failure; `cpu-native/` is
the successful run with access to Core ML's native temporary directories.

New runs require fresh output directories:

```sh
.venv/bin/python scripts/coreml/export_f32_512.py \
  --output-dir artifacts/coreml/f32-512/another-run
.venv/bin/python scripts/coreml/assess_f32.py \
  artifacts/coreml/f32-512/another-run/EmbeddingGemma2F32.mlpackage \
  --output-dir artifacts/coreml/f32-512/another-run/assessments/initial
```

## LiteRT-LM Comparison

The CPU/GPU bundle from
`litert-community/embeddinggemma-2-text-270m-litert-lm` is pinned to revision
`9be6e8b90982095dc05c2bd162e4b954ee4dbac7`. Its size is 164,626,432 bytes
(about 165 MB); both it and the downloaded model card match upstream hashes.
The model card describes int4 per-channel QAT for transformer and embeddings.

The optional `litert` dependencies lock LiteRT-LM/API/builder 0.18.1 and
SentencePiece 0.2.2. The comparison uses the actual LiteRT-LM EmbeddingEngine
on CPU with four threads, min/max input signatures fixed to 512, explicit BOS/EOS
insertion, 768-dimensional normalized output, and overflow set to ERROR.

The embedded tokenizer matches the original tokenizer.model hash and reproduces
every baseline token sequence exactly. Reconstructed right-padding masks match
the saved masks; the native runtime constructs its own mask from token length.
The Python API accepts text, so native input buffers are not directly supplied
or dumped. This interface limitation and the upstream mask implementation link
are recorded in the report. Unpacked bundle sections and their hashes are retained.

All 13 outputs were finite and unit-normalized (maximum norm error `5.14e-7`).
Short-fixture cosine against the padded float32 baseline ranged from `0.973561`
to `0.992890`; 511/512-token cosine was `0.958677`/`0.959037`. Maximum absolute
error was `0.043492`. None of the fixtures met the existing F32 profile.
Mars remained first (`0.880536`), but Saturn (`0.686730`) narrowly overtook
Venus (`0.685942`), reversing the baseline's lower two ranks. Six input-guard
checks and two native 513-token rejection checks passed. This is a descriptive
quantized-model comparison, not an approved lossy-quality or device-performance
claim. A broader retrieval evaluation would be needed to judge acceptability.

The report links the source receipt, bundle, baseline, script/lock snapshots,
verified inputs, and saved predictions. Core ML and original-model artifacts
remain unchanged. `inspection/` and `probe-cache/` contain preliminary bundle
inspection and one-query runtime checks, separate from the saved assessment.

```sh
uv sync --locked --extra litert
# Download only if the pinned local bundle and receipt do not already exist:
.venv/bin/python scripts/litert/fetch_model.py
.venv/bin/python scripts/litert/assess_litertlm.py \
  --output-dir artifacts/litert/text-270m/assessments/another-run
```
