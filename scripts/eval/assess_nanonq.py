"""Compare top-1/5/10 NanoNQ retrieval for the three saved 512-token models."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from transformers import AutoTokenizer

from fetch_nanonq import DATASET, FILES, REPOSITORY, REVISION

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/coreml"))
sys.path.insert(0, str(ROOT / "scripts/litert"))
from evidence import BASELINE, MODEL, RECEIPT, file_hashes, load_baseline, sha256, verify_local_model, write_report
from assess_litertlm import DESTINATION, FILENAME, verify_bundle

F32 = ROOT / "artifacts/coreml/f32-512-ios18/EmbeddingGemma2F32.mlpackage"
INT4 = ROOT / "artifacts/coreml/int4-512/EmbeddingGemma2Int4.mlpackage"
LITERT_ASSESSMENT = ROOT / "artifacts/litert/text-270m/assessments/initial/report.json"


def verify_dataset():
    receipt = json.loads((DATASET / "receipt.json").read_text())
    if receipt["repository"] != REPOSITORY or receipt["revision"] != REVISION:
        raise ValueError("Dataset identity differs from the NanoNQ pin")
    if {item["path"] for item in receipt["files"]} != set(FILES):
        raise ValueError("Unexpected dataset file inventory")
    for item in receipt["files"]:
        path = DATASET / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Dataset hash or size mismatch: {item['path']}")
    return receipt


def coreml_provenance(package, label, baseline_links):
    export_path = package.parent / "export-report.json"
    assessment_path = package.parent / "assessments/initial/report.json"
    export = json.loads(export_path.read_text())
    assessment = json.loads(assessment_path.read_text())
    hashes = file_hashes(package)
    if (not hashes or export["status"] != "exported" or export["package_name"] != package.name
            or hashes != export["package_hashes"] or hashes != assessment["package_hashes"]
            or sha256(export_path) != assessment["export_report_sha256"]):
        raise ValueError(f"Saved package provenance mismatch: {label}")
    if label == "coreml_f32":
        if assessment["status"] != "passed" or assessment["quality_profile"] != "f32-parity" or export["precision"] != "float32":
            raise ValueError("Float32 reference must have passed saved-package parity")
    elif assessment["status"] != "compared" or export["precision"] != "int4-weights":
        raise ValueError("Expected the previously compared Core ML int4 candidate")
    if sha256(assessment_path.parent / "predictions.npz") != assessment["predictions_sha256"]:
        raise ValueError(f"Prior prediction hash mismatch: {label}")
    for evidence in (export, assessment):
        for key, value in baseline_links.items():
            if key != "baseline_directory" and evidence[key] != value:
                raise ValueError(f"Baseline provenance mismatch for {label}: {key}")
    return {"package": str(package), "package_hashes": hashes, "export_report_sha256": sha256(export_path),
            "assessment_report_sha256": sha256(assessment_path), "assessment_path": str(assessment_path)}


def prepare(output, report):
    receipt = verify_dataset()
    _, _, _, _, _, baseline_links = load_baseline(BASELINE)
    verify_local_model(baseline_links["model_source_sha256"])
    verify_bundle()
    report.update(baseline_links)
    report["dataset"] = {**receipt, "receipt_sha256": sha256(DATASET / "receipt.json")}
    source = json.loads(RECEIPT.read_text())
    tokenizer_path = MODEL / "tokenizer.model"
    tokenizer_hash = next(item["sha256"] for item in source["files"] if item["path"] == "tokenizer.model")
    if sha256(tokenizer_path) != tokenizer_hash:
        raise ValueError("Original tokenizer hash mismatch")
    lite_prior = json.loads(LITERT_ASSESSMENT.read_text())
    if (lite_prior["status"] != "compared" or lite_prior["embedded_tokenizer_sha256"] != tokenizer_hash
            or lite_prior["bundle_sha256"] != sha256(DESTINATION / FILENAME)):
        raise ValueError("LiteRT bundle/tokenizer differs from the assessed bundle")
    report["models"] = {
        "coreml_f32": coreml_provenance(F32, "coreml_f32", baseline_links),
        "coreml_int4": coreml_provenance(INT4, "coreml_int4", baseline_links),
        "litert_int4": {"bundle": str(DESTINATION / FILENAME), "bundle_sha256": lite_prior["bundle_sha256"],
                        "receipt_sha256": lite_prior["litert_receipt_sha256"],
                        "prior_assessment_sha256": sha256(LITERT_ASSESSMENT), "tokenizer_sha256": tokenizer_hash},
    }
    prompts = json.loads((MODEL / "config_sentence_transformers.json").read_text())["prompts"]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    if tokenizer.padding_side != "right" or tokenizer.pad_token_id != 0:
        raise ValueError("Expected the pinned right-padding tokenizer")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    report["prompts"] = {name: prompts[name] for name in ("SearchQuery", "Document")}
    corpus = pq.read_table(DATASET / "corpus/train-00000-of-00001.parquet").to_pylist()
    queries = pq.read_table(DATASET / "queries/train-00000-of-00001.parquet").to_pylist()
    labels = pq.read_table(DATASET / "qrels/train-00000-of-00001.parquet").to_pylist()
    all_doc_ids, all_query_ids = {row["_id"] for row in corpus}, {row["_id"] for row in queries}
    if len(all_doc_ids) != len(corpus) or len(all_query_ids) != len(queries):
        raise ValueError("Duplicate dataset IDs")
    qrels = {query_id: set() for query_id in all_query_ids}
    for row in labels:
        if row["query-id"] not in all_query_ids or row["corpus-id"] not in all_doc_ids:
            raise ValueError("Relevance label refers to missing data")
        qrels[row["query-id"]].add(row["corpus-id"])
    exclusions, encoded = [], {}

    def eligible(rows, prompt_name, kind):
        selected = []
        for row in sorted(rows, key=lambda row: row["_id"]):
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                exclusions.append({"kind": kind, "id": row["_id"], "reason": "empty_text"})
                continue
            submitted = prompts[prompt_name] + text
            # Both tokenizers must agree on every evaluated sequence, not just the smoke fixtures.
            ids = tokenizer(submitted, truncation=False, add_special_tokens=True)["input_ids"]
            native_ids = [sp.bos_id(), *sp.encode(submitted), sp.eos_id()]
            if ids != native_ids:
                raise ValueError(f"Hugging Face/SentencePiece token mismatch: {kind} {row['_id']}")
            if len(ids) > 512:
                exclusions.append({"kind": kind, "id": row["_id"], "reason": "over_512_prompted_tokens", "token_count": len(ids)})
                continue
            record = {**row, "prompt_name": prompt_name, "submitted_text": submitted, "token_count": len(ids)}
            selected.append(record)
            encoded[(kind, row["_id"])] = ids
        return selected

    documents = eligible(corpus, "Document", "document")
    selected_queries = eligible(queries, "SearchQuery", "query")
    doc_ids = {row["_id"] for row in documents}
    kept_queries = []
    for row in selected_queries:
        if not qrels[row["_id"]] & doc_ids:
            exclusions.append({"kind": "query", "id": row["_id"], "reason": "no_remaining_relevant_document"})
        else:
            kept_queries.append(row)
    queries = kept_queries
    if not documents or not queries:
        raise ValueError("No eligible evaluation set remains")
    relevant = {row["_id"]: sorted(qrels[row["_id"]] & doc_ids) for row in queries}
    rows = documents + queries
    ids = np.zeros((len(rows), 512), dtype=np.int32)
    masks = np.zeros_like(ids)
    for index, row in enumerate(rows):
        kind = "document" if index < len(documents) else "query"
        tokens = encoded[(kind, row["_id"])]
        ids[index, :len(tokens)] = tokens
        masks[index, :len(tokens)] = 1
    np.savez_compressed(output / "inputs.npz", input_ids=ids, attention_mask=masks)
    records = {"documents": documents, "queries": queries, "relevance": relevant, "exclusions": exclusions}
    write_report(output / "records.json", records)
    report["coverage"] = {
        "original_documents": len(corpus), "evaluated_documents": len(documents),
        "original_queries": len(all_query_ids), "evaluated_queries": len(queries),
        "original_positive_labels": len(labels), "evaluated_positive_labels": sum(map(len, relevant.values())),
        "exclusion_counts": {reason: sum(row["reason"] == reason for row in exclusions) for reason in sorted({row["reason"] for row in exclusions})},
        "exclusions": exclusions,
    }
    report["inputs_sha256"] = sha256(output / "inputs.npz")
    report["records_sha256"] = sha256(output / "records.json")
    report["input_verification"] = "HF tokenizer IDs exactly equal embedded SentencePiece + BOS/EOS on every eligible record; fixed right masks. LiteRT native buffers remain runtime-managed, as in the prior assessment."
    print(f"Eligible set: {len(documents)}/{len(corpus)} passages; {len(queries)}/{len(all_query_ids)} questions; {sum(map(len, relevant.values()))}/{len(labels)} relevance labels", flush=True)
    return documents, queries, relevant, ids, masks


def validate_vectors(vectors):
    if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != 768 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite float32 768-dimensional embeddings")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4, rtol=0)


def embed(label, rows, ids, masks, output):
    import coremltools as ct
    import litert_lm
    start = time.monotonic()
    vectors = np.empty((len(rows), 768), dtype=np.float32)
    if label.startswith("coreml"):
        model = ct.models.MLModel(str(F32 if label == "coreml_f32" else INT4), compute_units=ct.ComputeUnit.CPU_ONLY)
        def predict(index):
            return model.predict({"input_ids": ids[index:index+1], "attention_mask": masks[index:index+1]})["embedding"]
    else:
        cache = output / "litert-runtime-cache"
        cache.mkdir()
        model = litert_lm.EmbeddingEngine(str(DESTINATION / FILENAME), backend=litert_lm.Backend.CPU(thread_count=4),
                                        cache_dir=str(cache), min_input_length=512, max_input_length=512)
        options = litert_lm.EmbeddingOptions(normalize=True, insert_special_tokens=True, output_size=768,
                                            input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR)
        def predict(index):
            return np.asarray(model.compute_embedding(rows[index]["submitted_text"], options=options).embedding, dtype=np.float32)[None]
    try:
        for index in range(len(rows)):
            vector = np.asarray(predict(index))
            if vector.shape != (1, 768):
                raise ValueError(f"Unexpected output shape for {label}")
            validate_vectors(vector)
            vectors[index] = vector[0]
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(f"{label}: {index+1}/{len(rows)} embedded ({time.monotonic()-start:.1f}s elapsed)", flush=True)
    finally:
        if label == "litert_int4":
            model.close()
    return vectors


def score(vectors, documents, queries, relevant):
    count = len(documents)
    scores = vectors[count:] @ vectors[:count].T
    doc_ids = [row["_id"] for row in documents]
    results = []
    for index, query in enumerate(queries):
        order = np.argsort(-scores[index], kind="stable")[:10]
        top = [doc_ids[i] for i in order]
        positive = set(relevant[query["_id"]])
        hits = {f"hit@{k}": bool(set(top[:k]) & positive) for k in (1, 5, 10)}
        results.append({"query_id": query["_id"], "query": query["text"], "relevant_ids": sorted(positive), **hits,
                        "top10": [{"document_id": doc_ids[i], "score": float(scores[index, i]), "relevant": doc_ids[i] in positive} for i in order]})
    metrics = {f"hit@{k}": {"count": sum(row[f"hit@{k}"] for row in results), "total": len(queries),
                             "rate": sum(row[f"hit@{k}"] for row in results) / len(queries)} for k in (1, 5, 10)}
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh evaluation directory")
    parser.add_argument("--prepare-only", action="store_true", help="Check provenance and input coverage without model inference")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "stage": "prepare", "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pyarrow", "transformers", "coremltools", "litert-lm-api", "sentencepiece")},
        "script_sha256": sha256(Path(__file__)), "fetch_script_sha256": sha256(Path(__file__).with_name("fetch_nanonq.py")),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"), "compute_units": "CPU_ONLY", "sequence_length": 512,
        "policy": "One shared eligible set; reject empty or over-512-token prompted inputs; omit queries with no remaining relevant passage; no truncation or chunking.",
        "metric": "Hit@k: fraction of questions with at least one labeled relevant passage in the top k; not document recall.",
        "search": "Exact dot products over normalized embeddings; corpus sorted by ID with stable tie breaking.",
        "limitations": "Small, possibly filtered English NanoNQ subset; descriptive retrieval comparison, not universal quality or device-performance evidence.",
    }
    try:
        for src in (Path(__file__), Path(__file__).with_name("fetch_nanonq.py"), ROOT / "scripts/coreml/evidence.py",
                    ROOT / "scripts/litert/assess_litertlm.py", ROOT / "scripts/litert/fetch_model.py", ROOT / "uv.lock"):
            shutil.copyfile(src, output / src.name)
        report["shared_code_hashes"] = {name: sha256(output / name) for name in ("evidence.py", "assess_litertlm.py", "fetch_model.py")}
        documents, queries, relevant, ids, masks = prepare(output, report)
        if args.prepare_only:
            report.update(status="prepared", stage="inputs_prepared")
            return
        rows = documents + queries
        report["metrics"], per_model = {}, {}
        for label in ("coreml_f32", "coreml_int4", "litert_int4"):
            report["stage"] = f"embed_{label}"
            vectors = embed(label, rows, ids, masks, output)
            path = output / f"{label}-embeddings.npz"
            np.savez_compressed(path, documents=vectors[:len(documents)], queries=vectors[len(documents):])
            metrics, rankings = score(vectors, documents, queries, relevant)
            report["metrics"][label] = {"metrics": metrics, "embeddings_sha256": sha256(path)}
            per_model[label] = rankings
            write_report(output / f"{label}-rankings.json", {"queries": rankings})
            report["metrics"][label]["rankings_sha256"] = sha256(output / f"{label}-rankings.json")
            print(f"{label} results: {metrics}", flush=True)
        disagreements = []
        doc_text = {row["_id"]: row["text"] for row in documents}
        for index, query in enumerate(queries):
            records = {label: results[index] for label, results in per_model.items()}
            if len({tuple(row[f"hit@{k}"] for k in (1, 5, 10)) for row in records.values()}) > 1:
                example = {"query_id": query["_id"], "query": query["text"], "relevant_ids": relevant[query["_id"]], "models": {}}
                for label, row in records.items():
                    example["models"][label] = {key: row[key] for key in ("hit@1", "hit@5", "hit@10")}
                    example["models"][label]["top3"] = [{**hit, "text_preview": doc_text[hit["document_id"]][:300]} for hit in row["top10"][:3]]
                disagreements.append(example)
        write_report(output / "disagreements.json", {"selection": "All Hit@1/5/10 disagreements in sorted query-ID order", "queries": disagreements})
        report["disagreements_sha256"] = sha256(output / "disagreements.json")
        report["disagreement_count"] = len(disagreements)
        if sha256(ROOT / "uv.lock") != report["uv_lock_sha256"] or verify_dataset() != {key: value for key, value in report["dataset"].items() if key != "receipt_sha256"}:
            raise ValueError("Dataset or environment lock changed during evaluation")
        for label, package in (("coreml_f32", F32), ("coreml_int4", INT4)):
            if coreml_provenance(package, label, load_baseline(BASELINE)[-1]) != report["models"][label]:
                raise ValueError(f"Model provenance changed during evaluation: {label}")
        verify_bundle()
        if sha256(DESTINATION / FILENAME) != report["models"]["litert_int4"]["bundle_sha256"]:
            raise ValueError("LiteRT bundle changed during evaluation")
        report.update(status="compared", stage="evaluation_complete")
        print(f"Completed NanoNQ retrieval comparison: {output}", flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_report(output / "report.json", report)


if __name__ == "__main__":
    main()
