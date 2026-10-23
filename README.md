# EmbeddingGemma 2 Text 270M → Core ML

Experiment to convert the 270M text component of Google's official
`embeddinggemma-2` checkpoint into a Core ML embedder for on-device retrieval.

Download the packages and shared tokenizer from
[Mannyking/embeddinggemma-2-text-270m-coreml](https://huggingface.co/Mannyking/embeddinggemma-2-text-270m-coreml).

## The Idea

Establish that the full embedding pipeline converts correctly, then check
whether a smaller quantized package retains useful retrieval quality.

```text
original model → baseline → F32 export → saved-package assessment
assessed F32 (iOS 18) → int4 compression → saved-package assessment
```

This experiment uses only the text model; vision and audio encoders are disabled.
The Core ML packages accept `(1, 512)` int32 token IDs and an attention mask,
and return one normalized 768-dimensional float32 embedding. The caller handles
query/document prompts, tokenization, and right padding. The 512-token limit
includes prompts and special tokens; longer inputs are rejected without truncation.

## Results and Evaluation

Both F32 packages passed CPU-only parity against 13 original-model fixtures,
including exact 511- and 512-token inputs. Maximum absolute error against the
padded F32 baseline was approximately `2.7e-7`.

The Core ML int4 package uses symmetric post-training weight compression in
blocks of 32, including attention weights; activations remain float32. I decided
to go full int4 on this version, as opposed to mixed int4/int8, to mimic the
equivalent LiteRT impl just as I did last time. (Their version is apparently
QAT this time so that's something to keep in mind).

### NanoNQ Retrieval

The three models searched the same 5,012 passages for 50 questions. Twenty-three
passages above 512 prompted tokens were excluded. Hit@k counts questions with at
least one labeled relevant passage in the first k results.

| Model | Saved size | Hit@1 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| Core ML F32 (iOS 18 source) | 1,090.2 MB | 35/50 | 44/50 | 45/50 |
| Core ML int4 | 175.9 MB | 33/50 | 43/50 | 46/50 |
| LiteRT-LM int4 | 164.6 MB | 34/50 | 43/50 | 44/50 |

This is a quick, single-dataset check for large retrieval quality regressions.
The takeaway should be all models appear capable.

## Platform Compatibility

The regular F32 package targets **iOS/iPadOS 15 or macOS 12 and newer**.
The int4 package requires **iOS/iPadOS 18 or macOS 15 and newer**.

Core ML's int4 compression requires a separately exported and assessed F32
package targeting iOS 18/macOS 15. That compression source is also the F32
reference used in NanoNQ, so in total there are 3 F32s - original, ios15 variant,
ios18 variant. Validation was performed on macOS CPU.

## Reproduce the Findings and Packages

Requires macOS, Python 3.11, and `uv`. Weights, datasets, packages, and generated
assessments are not in Git. Start with the pinned original-model smoke test,
then capture the baseline and export/assess F32:

```sh
uv sync --locked --extra litert --extra eval
.venv/bin/hf download google/embeddinggemma-2 \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --local-dir models/embeddinggemma-2

.venv/bin/python scripts/original/smoke_test_f32.py
.venv/bin/python scripts/original/capture_baseline_f32.py \
  --output artifacts/baseline-f32-512

.venv/bin/python scripts/coreml/export_f32_512.py \
  --output-dir artifacts/coreml/f32-512
.venv/bin/python scripts/coreml/assess_f32.py \
  artifacts/coreml/f32-512/EmbeddingGemma2F32.mlpackage \
  --output-dir artifacts/coreml/f32-512/assessments/initial
```

Use fresh output directories when repeating a run. Export, compression, and
assessment entrypoints provide `--help` for their options.

## Further Reading

- [CONTEXT.md](CONTEXT.md): workflow, local artifact map, and current limits.
- [AGENTS.md](AGENTS.md): working rules and evidence preservation.
- [Official checkpoint and model card](https://huggingface.co/google/embeddinggemma-2)
- [LiteRT-LM comparison model](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm)
- [NanoNQ dataset](https://huggingface.co/datasets/zeta-alpha-ai/NanoNQ)
- [Core ML Tools](https://github.com/apple/coremltools)

---

on the side, codex sure loves SHA evaluations/evidencing these days. I considered prohibiting to keep
things lean but then decided not to. Generation has become cheap, so evaluation on genuineness doesn't hurt, especially when it's probably a model that will perform that check again anyway
