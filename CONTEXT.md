# Working Context

Text-only EmbeddingGemma 2 (270M) Core ML experiment. Work only in this repo; v1 is
an independent project. Python 3.11 and dependencies are pinned in `uv.lock`.
Model weights, datasets, and generated artifacts are not version controlled.

## Workflow

- `scripts/original/`: smoke test and original-model baseline capture.
- `scripts/coreml/`: F32 export, int4 compression, saved-package assessment.
- `scripts/litert/`: pinned bundle download and baseline comparison.
- `scripts/eval/`: pinned NanoNQ download and retrieval evaluation.

With the local model available, start with:

```sh
uv sync --locked --extra litert --extra eval
.venv/bin/python scripts/original/smoke_test_f32.py
```

Model identities and hashes live in `models/receipts/`; dataset provenance is
in `datasets/nanonq/receipt.json`. Export and assessment options are documented
by their `--help` output.

## Local Artifacts

All paths are relative to the repository root. Availability on a new machine
must be checked; these directories are ignored by Git.

| Purpose | Location |
| --- | --- |
| Original checkpoint and LiteRT bundle | `models/` |
| Original-model reference inputs and outputs | `artifacts/baseline-f32-512/` |
| Regular F32 package, iOS 15/macOS 12 | `artifacts/coreml/f32-512/` |
| F32 compression source, iOS 18/macOS 15 | `artifacts/coreml/f32-512-ios18/` |
| Int4 package, iOS 18/macOS 15 | `artifacts/coreml/int4-512/` |
| LiteRT comparison | `artifacts/litert/text-270m/assessments/initial/` |
| NanoNQ results | `artifacts/evaluations/nanonq-512/initial/` |
| Earlier attempts and historical comparisons | `artifacts/archive/` |

Core ML assessments sit under each package's `assessments/initial/`. Reports,
predictions, hashes, and code snapshots contain the detailed evidence.
`artifacts/INDEX.md` provides local navigation. Historical paths inside reports
are unchanged; the local `artifacts/archive/RELOCATION.md` records their mapping,
including the former checkout name `embeddinggemma2-coreml`.

## Verified State and Limits

- Both F32 packages passed saved-artifact parity against 13 original-model
  fixtures on macOS CPU. The regular iOS 15 export matched its prior iOS 17
  outputs exactly on those fixtures.
- Int4 compression requires the separately assessed iOS 18 F32 source. It uses
  int4 weights with float32 activations; the package is about 176 MB.
- Core ML int4 and LiteRT were compared with the baseline and evaluated for
  retrieval. Neither meets F32 parity; no lossy acceptance profile is approved.
- A single-projection feasibility probe preserved LiteRT's exact int4 weights
  and scales in Core ML. Full-model QAT transfer remains unimplemented;
  exported scaling differences and the remaining parameters need mapping.
- NanoNQ used the iOS 18 F32 source, Core ML int4, and LiteRT on 50 questions
  and 5,012 passages. Its 512-token filter excluded 23 passages, removing no
  labeled relevant passages. Scores remain unchanged after the iOS 15 export.
- Core ML takes `(1, 512)` int32 token IDs and masks and returns a normalized
  `(1, 768)` float32 embedding. Prompting/tokenization are external. LiteRT-LM
  accepts prompted text; native token/mask buffers are runtime-managed.
- Evidence is text-only and CPU-only. The small filtered retrieval test does
  not establish general superiority, study-app quality, or device performance.
