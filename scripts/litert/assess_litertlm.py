"""Compare the pinned LiteRT-LM int4 text bundle with the saved float32 baseline."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import litert_lm
from litert_lm_builder import unpack_litertlm_file
from litert_lm_builder.runtime.proto import embedding_metadata_pb2
from google.protobuf import text_format
import numpy as np
import sentencepiece as spm

from fetch_model import DESTINATION, FILENAME, RECEIPT, REPOSITORY, REVISION

# Reuse baseline/hash checks without importing Core ML or the original model.
COREML_SCRIPTS = Path(__file__).resolve().parents[1] / "coreml"
sys.path.insert(0, str(COREML_SCRIPTS))
from evidence import BASELINE, F32_PARITY, ROOT, compare, file_hashes, load_baseline, sha256, write_report

RUNTIME_VERSION = "0.18.1"
MASK_SOURCE = "https://github.com/google-ai-edge/LiteRT-LM/blob/v0.18.1/runtime/executor/embedding_litert_compiled_model_executor.cc#L406-L423"


def verify_bundle():
    source = json.loads(RECEIPT.read_text())
    if source["repository"] != REPOSITORY or source["revision"] != REVISION:
        raise ValueError("Unexpected LiteRT-LM source identity")
    for entry in source["files"]:
        path = DESTINATION / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"LiteRT-LM source hash/size mismatch: {entry['path']}")
    if FILENAME not in {entry["path"] for entry in source["files"]}:
        raise ValueError("Receipt does not cover the selected bundle")
    return source


def checked_tokens(sp, text, prompt):
    if not text.strip():
        raise ValueError("Empty or whitespace-only content is rejected before prompting")
    tokens = [sp.bos_id(), *sp.encode(prompt + text), sp.eos_id()]
    if len(tokens) > 512:
        raise ValueError(f"Input has {len(tokens)} prompted tokens; selected signature limit is 512")
    return tokens


def exact_over_limit(sp, prompt):
    for count in range(449, 514):
        text = "planet " * count
        if len(sp.encode(prompt + text)) + 2 == 513:
            return text
    raise ValueError("Could not construct a real 513-token input")


def run(output, baseline, report):
    if importlib.metadata.version("litert-lm-api") != RUNTIME_VERSION:
        raise ValueError("Use the locked LiteRT-LM 0.18.1 runtime")
    source = verify_bundle()
    metadata, names, inputs, expected, unpadded, links = load_baseline(baseline)
    report.update(links)
    report.update(litert_repository=source["repository"], litert_revision=source["revision"],
                  litert_receipt_sha256=sha256(RECEIPT), bundle_sha256=sha256(DESTINATION / FILENAME),
                  bundle_size_bytes=(DESTINATION / FILENAME).stat().st_size, fixture_order=names)
    shutil.copyfile(RECEIPT, output / "litert-source.json")
    report["stage"] = "verify_tokenizer_and_inputs"
    unpacked = output / "unpacked"
    unpacked.mkdir()
    unpack_litertlm_file(str(DESTINATION / FILENAME), str(unpacked))
    report["unpacked_hashes"] = file_hashes(unpacked)
    original_source = json.loads((baseline / "model-source.json").read_text())
    tokenizer = next(unpacked.glob("*_SP_Tokenizer.spiece"))
    original_tokenizer_hash = next(item["sha256"] for item in original_source["files"] if item["path"] == "tokenizer.model")
    if sha256(tokenizer) != original_tokenizer_hash:
        raise ValueError("Embedded tokenizer differs from the pinned original tokenizer")
    report["embedded_tokenizer_sha256"] = original_tokenizer_hash
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    embedding_metadata = embedding_metadata_pb2.EmbeddingMetadata()
    text_format.Parse((unpacked / "EmbeddingMetadataProto.pbtext").read_text(), embedding_metadata)
    if (list(embedding_metadata.bos_token.token_ids.ids) != [sp.bos_id()]
            or list(embedding_metadata.eos_token.token_ids.ids) != [sp.eos_id()]):
        raise ValueError("Bundle special-token insertion differs from expected BOS/EOS")
    report["special_tokens"] = {"bos": sp.bos_id(), "eos": sp.eos_id()}
    submitted, input_arrays = [], {}
    for case, (ids, mask) in zip(metadata["cases"], inputs):
        tokens = checked_tokens(sp, case["text"], case["prompt"])
        count = len(tokens)
        if count != case["token_count"] or tokens != ids[0, :count].tolist():
            raise ValueError(f"Embedded tokenizer does not reproduce baseline IDs: {case['id']}")
        runtime_mask = np.zeros((1, 512), dtype=np.int32)
        runtime_mask[:, :count] = 1
        np.testing.assert_array_equal(runtime_mask, mask)
        input_arrays[case["id"] + "__input_ids"] = ids
        input_arrays[case["id"] + "__attention_mask"] = runtime_mask
        submitted.append(case["prompt"] + case["text"])
    np.savez_compressed(output / "verified-inputs.npz", **input_arrays)
    report["verified_inputs_sha256"] = sha256(output / "verified-inputs.npz")
    report["input_verification"] = {
        "token_ids": "Embedded SentencePiece model plus metadata BOS/EOS exactly reproduces every saved unpadded token sequence.",
        "attention_mask": "Reconstructed 512-token right masks match baseline. Native runtime fills input_mask with 1 for token count, then 0 (source below).",
        "native_mask_source": MASK_SOURCE,
        "limitation": "Python embedding API accepts text; native token and mask buffers are not directly supplied or dumped. Padding embeddings are runtime-managed.",
    }
    with litert_lm.ModelInfo(str(DESTINATION / FILENAME)) as info:
        if not info.is_embedding_model or info.embedding.dimension != 768 or 512 not in info.embedding.signature_selection:
            raise ValueError("Expected a 768-dimensional embedding model with a 512-token signature")
        report["model_info"] = {"dimension": info.embedding.dimension, "signatures": info.embedding.signature_selection,
                                "min_runtime_version": info.min_runtime_version}
    report["stage"] = "compare_saved_bundle"
    predictions, records = {}, []
    options = litert_lm.EmbeddingOptions(
        normalize=True, insert_special_tokens=True, output_size=768,
        input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR,
    )
    cache = output / "runtime-cache"
    cache.mkdir()
    with litert_lm.EmbeddingEngine(
        str(DESTINATION / FILENAME), backend=litert_lm.Backend.CPU(thread_count=4),
        min_input_length=512, max_input_length=512, cache_dir=str(cache),
    ) as engine:
        for index, (name, text) in enumerate(zip(names, submitted)):
            response = engine.compute_embedding(text, options=options)
            actual = np.asarray(response.embedding, dtype=np.float32)[None, :]
            padded_metrics, padded_parity = compare(actual, expected[index:index+1])
            unpadded_metrics, unpadded_parity = compare(actual, unpadded[index:index+1])
            if abs(padded_metrics["norm"] - 1) > 1e-4:
                raise ValueError(f"Non-unit LiteRT-LM output: {name}")
            predictions[name] = actual
            records.append({"id": name, "submitted_text": text, "token_count": metadata["cases"][index]["token_count"],
                            "padded_baseline": padded_metrics, "unpadded_baseline": unpadded_metrics,
                            "f32_parity": padded_parity and unpadded_parity})
            report["cases"] = records
            print(f"{name}: cosine {padded_metrics['cosine']:.6f}; max error {padded_metrics['max_absolute_error']:.6f}", flush=True)
        rejected = []
        prompts = {case["prompt_name"]: case["prompt"] for case in metadata["cases"]}
        for prompt_name, prompt in prompts.items():
            over = exact_over_limit(sp, prompt)
            for name, text in (("empty", ""), ("whitespace", " \t\n"), ("over_limit", over)):
                try:
                    checked_tokens(sp, text, prompt)
                except ValueError as error:
                    rejected.append({"id": name, "prompt_name": prompt_name, "reason": str(error)})
                else:
                    raise ValueError(f"Input guard failed: {name}")
            # Also prove the native runtime rejects overflow, rather than chunking/truncating.
            try:
                engine.compute_embedding(prompt + over, options=options)
            except RuntimeError as error:
                rejected.append({"id": "native_513_tokens", "prompt_name": prompt_name, "reason": str(error)})
            else:
                raise ValueError("Native runtime accepted a 513-token input at max_input_length=512")
        report["rejected"] = rejected
    np.savez_compressed(output / "predictions.npz", **predictions)
    report["predictions_sha256"] = sha256(output / "predictions.npz")
    documents = metadata["retrieval"]["documents"]
    scores = np.concatenate([predictions[name] for name in documents]) @ predictions["query_mars"][0]
    ranking = [documents[i] for i in np.argsort(-scores)]
    report["retrieval"] = {"documents": documents, "scores": scores.tolist(), "ranking": ranking,
                            "baseline_ranking": metadata["retrieval"]["ranking"],
                            "ranking_unchanged": ranking == metadata["retrieval"]["ranking"]}
    report["summary"] = {
        "minimum_cosine": min(case["padded_baseline"]["cosine"] for case in records),
        "maximum_cosine": max(case["padded_baseline"]["cosine"] for case in records),
        "maximum_absolute_error": max(case["padded_baseline"]["max_absolute_error"] for case in records),
        "f32_parity": all(case["f32_parity"] for case in records) and report["retrieval"]["ranking_unchanged"],
    }
    if verify_bundle() != source or sha256(RECEIPT) != report["litert_receipt_sha256"]:
        raise ValueError("LiteRT-LM source changed during assessment")
    if load_baseline(baseline)[-1] != links or sha256(ROOT / "uv.lock") != report["uv_lock_sha256"]:
        raise ValueError("Baseline or environment lock changed during assessment")
    report.update(status="compared", stage="comparison_complete")
    print(f"Comparison saved: {output}\n{report['summary']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh comparison directory")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "stage": "verify_inputs", "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_type": "descriptive quantized-bundle comparison; no approved lossy acceptance profile",
        "backend": "CPU", "thread_count": 4, "min_input_length": 512, "max_input_length": 512,
        "normalize": True, "insert_special_tokens": True, "output_size": 768, "overflow_strategy": "ERROR",
        "f32_parity_diagnostic_thresholds": F32_PARITY,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("litert-lm", "litert-lm-api", "litert-lm-builder", "sentencepiece", "numpy")},
        "script_sha256": sha256(Path(__file__)), "evidence_script_sha256": sha256(COREML_SCRIPTS / "evidence.py"),
        "fetch_script_sha256": sha256(Path(__file__).with_name("fetch_model.py")), "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "limitations": "13 text fixtures on macOS CPU only; not a retrieval benchmark, latency study, or device assessment.",
    }
    try:
        for src in (Path(__file__), Path(__file__).with_name("fetch_model.py"), COREML_SCRIPTS / "evidence.py", ROOT / "uv.lock"):
            shutil.copyfile(src, output / src.name)
        run(output, args.baseline.expanduser().resolve(), report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_report(output / "report.json", report)


if __name__ == "__main__":
    main()
