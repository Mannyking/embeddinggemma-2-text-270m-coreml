"""Evaluate the saved v1 int4/int8 package on a completed v2 NanoNQ run's exact records."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from transformers import AutoTokenizer

from assess_nanonq import ROOT, file_hashes, score, sha256, validate_vectors, verify_dataset, write_report

SISTER = ROOT.parent / "embeddinggemma-coreml"
SOURCE = SISTER / "models/embeddinggemma-300m"
RECEIPT = SISTER / "models/receipts/embeddinggemma-300m.json"
PACKAGE = SISTER / "artifacts/coreml/int4-512-attention-int8/EmbeddingGemmaInt4AttentionInt8.mlpackage"
REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"


def provenance():
    receipt = json.loads(RECEIPT.read_text())
    if receipt["repository"] != "google/embeddinggemma-300m" or receipt["revision"] != REVISION:
        raise ValueError("Unexpected v1 source identity")
    for item in receipt["files"]:
        path = SOURCE / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"V1 source mismatch: {item['path']}")
    export_path = PACKAGE.parent / "export-report.json"
    assessment_path = PACKAGE.parent / "assessments/descriptive-initial-unsandboxed/report.json"
    baseline_path = SISTER / "artifacts/baseline-f32/metadata.json"
    export = json.loads(export_path.read_text())
    assessment = json.loads(assessment_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    hashes = file_hashes(PACKAGE)
    if (export["status"] != "exported" or export["precision"] != "int4-attention-int8"
            or assessment["status"] != "assessed" or assessment["comparison_mode"] != "descriptive"
            or export["revision"] != REVISION or assessment["baseline_revision"] != REVISION
            or baseline["revision"] != REVISION or baseline["model_manifest_sha256"] != sha256(RECEIPT)
            or not hashes or hashes != export["package_hashes"] or hashes != assessment["package_hashes"]
            or sha256(export_path) != assessment["export_report_sha256"]
            or sha256(baseline_path) != export["baseline_metadata_sha256"]
            or sha256(baseline_path) != assessment["baseline_metadata_sha256"]
            or sha256(Path(export["source_package"]).parent / "padded-fixtures.npz") != export["padded_fixtures_sha256"]
            or export["padded_fixtures_sha256"] != assessment["padded_fixtures_sha256"]):
        raise ValueError("V1 saved-package evidence mismatch")
    source = Path(export["source_package"])
    source_assessment = Path(export["source_assessment"])
    parent_export = json.loads((source.parent / "export-report.json").read_text())
    parent_assessment = json.loads(source_assessment.read_text())
    if (file_hashes(source) != export["source_package_hashes"]
            or parent_export["package_hashes"] != export["source_package_hashes"]
            or parent_assessment["package_hashes"] != export["source_package_hashes"]
            or sha256(source.parent / "export-report.json") != export["source_export_report_sha256"]
            or sha256(source_assessment) != export["source_assessment_sha256"]
            or parent_assessment["status"] != "passed" or parent_assessment["quality_profile"] != "f32-parity"):
        raise ValueError("V1 compression source evidence mismatch")
    return {"package": str(PACKAGE), "precision": "int4-attention-int8", "package_hashes": hashes, "revision": REVISION,
            "receipt_sha256": sha256(RECEIPT), "export_report_sha256": sha256(export_path),
            "assessment_report_sha256": sha256(assessment_path), "baseline_metadata_sha256": sha256(baseline_path)}


def verify_reference(reference):
    report = json.loads((reference / "report.json").read_text())
    if report["status"] != "compared" or report["stage"] != "evaluation_complete":
        raise ValueError("V2 reference must be a completed comparison")
    dataset = verify_dataset()
    if {k: v for k, v in report["dataset"].items() if k != "receipt_sha256"} != dataset:
        raise ValueError("V2 dataset differs from the pinned local dataset")
    for name, key in (("records.json", "records_sha256"), ("inputs.npz", "inputs_sha256")):
        if sha256(reference / name) != report[key]:
            raise ValueError(f"V2 reference mismatch: {name}")
    records = json.loads((reference / "records.json").read_text())
    for label, result in report["metrics"].items():
        path = reference / f"{label}-embeddings.npz"
        if sha256(path) != result["embeddings_sha256"] or sha256(reference / f"{label}-rankings.json") != result["rankings_sha256"]:
            raise ValueError(f"V2 result hash mismatch: {label}")
        with np.load(path, allow_pickle=False) as saved:
            if saved["documents"].shape != (len(records["documents"]), 768) or saved["queries"].shape != (len(records["queries"]), 768):
                raise ValueError("V2 embedding record count mismatch")
            vectors = np.concatenate((saved["documents"], saved["queries"]))
        validate_vectors(vectors)
        metrics, rankings = score(vectors, records["documents"], records["queries"], records["relevance"])
        if metrics != result["metrics"] or rankings != json.loads((reference / f"{label}-rankings.json").read_text())["queries"]:
            raise ValueError(f"V2 saved retrieval result mismatch: {label}")
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    reference = args.reference_run.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
              "reference_run": str(reference), "compute_units": "CPU_ONLY", "sequence_length": 512,
              "limitations": "Small filtered English NanoNQ comparison; v1 int4/int8 versus v2 variants; no device or performance claim."}
    with (output / "progress.log").open("x", buffering=1) as log:
        def progress(message):
            print(message, flush=True)
            print(message, file=log, flush=True)
        try:
            for path in (Path(__file__), Path(__file__).with_name("assess_nanonq.py"),
                         Path(__file__).with_name("fetch_nanonq.py"), ROOT / "scripts/coreml/evidence.py",
                         ROOT / "scripts/litert/assess_litertlm.py", ROOT / "scripts/litert/fetch_model.py", ROOT / "uv.lock"):
                shutil.copyfile(path, output / path.name)
            report["code_and_lock_hashes"] = {k: v for k, v in file_hashes(output).items() if k != "progress.log"}
            report["model"] = provenance()
            prior, records = verify_reference(reference)
            report["reference_report_sha256"] = sha256(reference / "report.json")
            report["reference_records_sha256"] = sha256(reference / "records.json")
            report["coverage"] = prior["coverage"]
            shutil.copyfile(reference / "records.json", output / "records.json")
            shutil.copyfile(RECEIPT, output / "model-source.json")
            tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True, trust_remote_code=False)
            source_prompts = json.loads((SOURCE / "config_sentence_transformers.json").read_text())["prompts"]
            prompts = {"SearchQuery": source_prompts["query"], "Document": source_prompts["document"]}
            if tokenizer.padding_side != "right" or tokenizer.pad_token_id != 0:
                raise ValueError("V1 tokenizer must use right padding with PAD 0")
            rows = records["documents"] + records["queries"]
            ids = np.zeros((len(rows), 512), dtype=np.int32)
            masks = np.zeros_like(ids)
            lengths = []
            for index, row in enumerate(rows):
                if not row["text"].strip():
                    raise ValueError("Empty reference text")
                submitted = prompts[row["prompt_name"]] + row["text"]
                if submitted != row["submitted_text"]:
                    raise ValueError("V1/v2 task prompts differ")
                tokens = tokenizer(submitted, add_special_tokens=True, truncation=False)["input_ids"]
                if not 0 < len(tokens) <= 512:
                    raise ValueError(f"V1 cannot accept shared record {row['_id']}: {len(tokens)} tokens")
                ids[index, :len(tokens)] = tokens
                masks[index, :len(tokens)] = 1
                lengths.append(len(tokens))
            np.savez_compressed(output / "inputs.npz", input_ids=ids, attention_mask=masks)
            report.update(inputs_sha256=sha256(output / "inputs.npz"), prompts={k: prompts[k] for k in ("SearchQuery", "Document")},
                          maximum_token_count=max(lengths), input_policy="Same v2 texts, order, relevance labels and prompts; v1's own tokenizer; no truncation or further exclusions.")
            progress(f"Verified shared set: {len(records['documents'])} passages, {len(records['queries'])} questions; v1 max {max(lengths)} tokens")
            if args.prepare_only:
                report["status"] = "prepared"
                return
            import coremltools as ct
            model = ct.models.MLModel(str(PACKAGE), compute_units=ct.ComputeUnit.CPU_ONLY)
            vectors = np.empty((len(rows), 768), dtype=np.float32)
            started = time.monotonic()
            for index in range(len(rows)):
                vector = np.asarray(model.predict({"input_ids": ids[index:index+1], "attention_mask": masks[index:index+1]})["embedding"])
                if vector.shape != (1, 768):
                    raise ValueError("Unexpected v1 output shape")
                validate_vectors(vector)
                vectors[index] = vector[0]
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    progress(f"v1_coreml_int4: {index+1}/{len(rows)} embedded ({time.monotonic()-started:.1f}s elapsed)")
            count = len(records["documents"])
            np.savez_compressed(output / "v1_coreml_int4-embeddings.npz", documents=vectors[:count], queries=vectors[count:])
            metrics, rankings = score(vectors, records["documents"], records["queries"], records["relevance"])
            write_report(output / "v1_coreml_int4-rankings.json", {"queries": rankings})
            report["metrics"] = {**{k: v["metrics"] for k, v in prior["metrics"].items()}, "v1_coreml_int4": metrics}
            report["embeddings_sha256"] = sha256(output / "v1_coreml_int4-embeddings.npz")
            report["rankings_sha256"] = sha256(output / "v1_coreml_int4-rankings.json")
            if provenance() != report["model"] or sha256(reference / "report.json") != report["reference_report_sha256"] or sha256(reference / "records.json") != report["reference_records_sha256"]:
                raise ValueError("Source evidence changed during evaluation")
            for name, expected in report["code_and_lock_hashes"].items():
                if sha256(output / name) != expected:
                    raise ValueError("Code snapshot changed")
            if sha256(ROOT / "uv.lock") != report["code_and_lock_hashes"]["uv.lock"]:
                raise ValueError("Environment lock changed during evaluation")
            report["status"] = "compared"
            progress(f"V1 completed: {metrics}")
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            progress(report["error"])
            raise
        finally:
            write_report(output / "report.json", report)


if __name__ == "__main__":
    main()
