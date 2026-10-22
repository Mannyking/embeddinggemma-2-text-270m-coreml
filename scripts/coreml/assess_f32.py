"""Assess saved-package float32 parity, or describe a quantized comparison on CPU."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

import coremltools as ct
import numpy as np

from evidence import (BASELINE, F32_PARITY, ROOT, compare, environment, file_hashes,
                      load_baseline, sha256, write_report)


def run(package, baseline, output, report, comparison_only=False):
    metadata, names, inputs, expected, unpadded, links = load_baseline(baseline)
    report.update(links)
    export_path = package.parent / "export-report.json"
    export = json.loads(export_path.read_text())
    if export["status"] != "exported" or export["package_name"] != package.name:
        raise ValueError("Expected a successfully exported package")
    if not comparison_only and export["precision"] != "float32":
        raise ValueError("F32 parity assessment requires a float32 export; use --comparison-only for lossy weights")
    for key, value in links.items():
        if key != "baseline_directory" and export[key] != value:
            raise ValueError(f"Export/baseline provenance mismatch: {key}")
    if export["fixture_order"] != names:
        raise ValueError("Export fixture order differs from the baseline")
    hashes = file_hashes(package)
    if not hashes or hashes != export["package_hashes"]:
        raise ValueError("Saved package inventory/hash differs from its export report")
    report.update(package_hashes=hashes, export_report_sha256=sha256(export_path), fixture_order=names)
    report["export_precision"] = export["precision"]
    report["stage"] = "load_saved_package"
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    report["stage"] = "artifact_comparison" if comparison_only else "artifact_parity"
    report["cases"] = []
    predictions, passed = {}, True
    for index, (name, (ids, mask)) in enumerate(zip(names, inputs)):
        actual = np.asarray(model.predict({"input_ids": ids, "attention_mask": mask})["embedding"])
        padded_metrics, padded_ok = compare(actual, expected[index:index+1])
        unpadded_metrics, unpadded_ok = compare(actual, unpadded[index:index+1])
        case_passed = padded_ok and unpadded_ok
        passed &= case_passed
        report["cases"].append({"id": name, "passed": case_passed, "padded_baseline": padded_metrics,
                                "unpadded_baseline": unpadded_metrics})
        predictions[name] = actual.copy()
        label = f"cosine {padded_metrics['cosine']:.6f}" if comparison_only else ('PASS' if case_passed else 'FAIL')
        print(f"{name}: {label}, max error {padded_metrics['max_absolute_error']:.3g}", flush=True)
    retrieval = metadata["retrieval"]
    documents = retrieval["documents"]
    scores = np.concatenate([predictions[name] for name in documents]) @ predictions["query_mars"][0]
    ranking = [documents[index] for index in np.argsort(-scores)]
    ranking_passed = ranking == retrieval["ranking"]
    report["retrieval"] = {"documents": documents, "scores": scores.tolist(), "ranking": ranking,
                            "expected_ranking": retrieval["ranking"], "passed": ranking_passed}
    np.savez_compressed(output / "predictions.npz", **predictions)
    report["predictions_sha256"] = sha256(output / "predictions.npz")
    if file_hashes(package) != hashes or sha256(export_path) != report["export_report_sha256"]:
        raise ValueError("Package or export report changed during assessment")
    if load_baseline(baseline)[-1] != links:
        raise ValueError("Baseline changed during assessment")
    report["f32_parity_passed"] = bool(passed and ranking_passed)
    report["summary"] = {
        "minimum_cosine": min(case["padded_baseline"]["cosine"] for case in report["cases"]),
        "maximum_cosine": max(case["padded_baseline"]["cosine"] for case in report["cases"]),
        "maximum_absolute_error": max(case["padded_baseline"]["max_absolute_error"] for case in report["cases"]),
        "maximum_norm_error": max(abs(case["padded_baseline"]["norm"] - 1) for case in report["cases"]),
    }
    if comparison_only:
        report.update(status="compared", stage="comparison_complete")
        print(f"Saved-package comparison recorded for {len(names)} fixtures (CPU_ONLY). Report: {output}", flush=True)
        return
    if not passed or not ranking_passed:
        raise ValueError("Saved package failed the fixed f32-parity profile; see report.json")
    report.update(status="passed", stage="assessment_complete")
    print(f"Saved package passed f32-parity on {len(names)} fixtures (CPU_ONLY). Report: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh assessment directory")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--comparison-only", action="store_true", help="Record lossy-model drift without claiming quality acceptance")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "stage": "verify_inputs",
        **environment(), "compute_units": "CPU_ONLY",
        "quality_profile": None if args.comparison_only else "f32-parity", "thresholds": F32_PARITY,
        "comparison_only": args.comparison_only,
        "acceptance_note": "No approved lossy-quality profile" if args.comparison_only else "Fixed F32 parity profile",
        "package": str(args.package.expanduser().resolve()), "script_sha256": sha256(Path(__file__)),
        "evidence_script_sha256": sha256(Path(__file__).with_name("evidence.py")),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "limitations": "Recorded 512-token text fixtures on macOS CPU only; no device, performance, or accelerator assessment.",
    }
    try:
        for src in (Path(__file__), Path(__file__).with_name("evidence.py"), ROOT / "uv.lock"):
            shutil.copyfile(src, output / src.name)
        run(args.package.expanduser().resolve(), args.baseline.expanduser().resolve(), output, report,
            comparison_only=args.comparison_only)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_report(output / "report.json", report)


if __name__ == "__main__":
    main()
