"""Compress an assessed iOS 18 float32 package to int4 weights; assess separately."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.proto import MIL_pb2

from evidence import (BASELINE, ROOT, environment, file_hashes, load_baseline,
                      sha256, verify_local_model, write_report)


def verify_source(package, assessment_path, baseline, report):
    metadata, names, _, _, _, links = load_baseline(baseline)
    verify_local_model(links["model_source_sha256"])
    source_hashes = file_hashes(package)
    export_path = package.parent / "export-report.json"
    export = json.loads(export_path.read_text())
    assessment = json.loads(assessment_path.read_text())
    if (export["status"] != "exported" or export["precision"] != "float32"
            or export["minimum_deployment_target"] != "iOS18/macOS15"
            or export["input_shape"] != [1, 512] or export["package_name"] != package.name):
        raise ValueError("Source must be a complete iOS 18 float32 512-token export")
    if (assessment["status"] != "passed" or assessment["quality_profile"] != "f32-parity"
            or len(assessment["cases"]) != len(names) or not all(case["passed"] for case in assessment["cases"])
            or not assessment["retrieval"]["passed"]):
        raise ValueError("Source must pass saved-package F32 parity on every fixture before compression")
    if not source_hashes or source_hashes != export["package_hashes"] or source_hashes != assessment["package_hashes"]:
        raise ValueError("Source package differs from the exact exported and assessed artifact")
    if assessment["export_report_sha256"] != sha256(export_path):
        raise ValueError("Source assessment links a different export report")
    if sha256(assessment_path.parent / "predictions.npz") != assessment["predictions_sha256"]:
        raise ValueError("Source assessment predictions have changed")
    for evidence in (export, assessment):
        for key, value in links.items():
            if key != "baseline_directory" and evidence[key] != value:
                raise ValueError(f"Source evidence uses a different baseline: {key}")
        if evidence["fixture_order"] != names:
            raise ValueError("Source evidence fixture order differs from the baseline")
    report.update(links)
    report.update(
        revision=metadata["revision"], fixture_order=names,
        source_package=str(package), source_package_hashes=source_hashes,
        source_export_report_sha256=sha256(export_path), source_assessment=str(assessment_path),
        source_assessment_sha256=sha256(assessment_path), source_predictions_sha256=assessment["predictions_sha256"],
    )
    return source_hashes


def operations(spec):
    if spec.WhichOneof("Type") != "mlProgram" or spec.specificationVersion < 9:
        raise ValueError("Expected an iOS 18 (specification version 9 or later) ML Program")
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            yield from block.operations


def audit_int4(spec, expected_counts):
    ops = list(operations(spec))
    producers = {output.name: (op, output) for op in ops for output in op.outputs}
    audited = []
    for op in ops:
        if op.type not in ("linear", "gather"):
            continue
        binding = "weight" if op.type == "linear" else "x"
        weight_name = op.inputs[binding].arguments[0].name
        dequant, _ = producers[weight_name]
        if dequant.type != "constexpr_blockwise_shift_scale":
            raise ValueError(f"Uncompressed {op.type} weight: {weight_name}")
        data = dequant.inputs["data"].arguments[0]
        if data.name:
            _, quantized = producers[data.name]
            data_type = quantized.type
        else:
            # Compressed payloads may be bound directly as typed blob values,
            # rather than produced by a separately named const operation.
            data_type = data.value.type
        dtype = MIL_pb2.DataType.Name(data_type.tensorType.dataType)
        if dtype != "INT4":
            raise ValueError(f"Expected INT4 data, got {dtype}: {weight_name}")
        audited.append({"op_type": op.type, "weight": weight_name,
                        "quantized_data": data.name or "inline_typed_value", "dtype": dtype})
    if Counter(item["op_type"] for item in audited) != expected_counts:
        raise ValueError("Compressed linear/gather inventory differs from the source")
    return audited


def run(source, source_assessment, baseline, output, report):
    source_hashes = verify_source(source, source_assessment, baseline, report)
    for path, name in ((source.parent / "export-report.json", "source-export-report.json"),
                       (source_assessment, "source-assessment.json")):
        shutil.copyfile(path, output / name)
    spec = ct.models.utils.load_spec(str(source))
    counts = Counter(op.type for op in operations(spec) if op.type in ("linear", "gather"))
    if counts["linear"] == 0 or counts["gather"] != 1:
        raise ValueError(f"Unexpected text-model compression targets: {counts}")
    report["target_operation_counts"] = dict(counts)
    report["stage"] = "compress_weights"
    model = ct.models.MLModel(str(source), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    int4 = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32, weight_threshold=2048
    )
    config = cto.OptimizationConfig(op_type_configs={"linear": int4, "gather": int4})
    print(f"Compressing {dict(counts)} weights to int4 (block size 32)", flush=True)
    compressed = cto.linear_quantize_weights(model, config)
    report["stage"] = "audit_compression"
    report["compressed_weights"] = audit_int4(compressed.get_spec(), counts)
    package = output / "EmbeddingGemma2Int4.mlpackage"
    compressed.save(str(package))
    # Inspect the saved artifact again, rather than relying only on the in-memory graph.
    if audit_int4(ct.models.utils.load_spec(str(package)), counts) != report["compressed_weights"]:
        raise ValueError("Saved compression graph differs from the audited graph")
    report.update(
        package_name=package.name, package_hashes=file_hashes(package),
        package_size_bytes=sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
        specification_version=compressed.get_spec().specificationVersion,
    )
    if file_hashes(source) != source_hashes or sha256(source_assessment) != report["source_assessment_sha256"]:
        raise ValueError("Source package or assessment changed during compression")
    if sha256(source.parent / "export-report.json") != report["source_export_report_sha256"]:
        raise ValueError("Source export report changed during compression")
    verify_local_model(report["model_source_sha256"])
    if load_baseline(baseline)[-1]["baseline_metadata_sha256"] != report["baseline_metadata_sha256"]:
        raise ValueError("Baseline changed during compression")
    report.update(status="exported", stage="compression_complete", artifact_parity="Unassessed; use assess_f32.py --comparison-only")
    print(f"Compressed package saved: {package} ({report['package_size_bytes']:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-assessment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh compression output directory")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "stage": "verify_source",
        **environment(), "precision": "int4-weights", "compute_precision": "float32",
        "input_dtype": "int32", "input_shape": [1, 512], "output_dtype": "float32", "output_shape": [1, 768],
        "minimum_deployment_target": "iOS18/macOS15",
        "script_sha256": sha256(Path(__file__)), "evidence_script_sha256": sha256(Path(__file__).with_name("evidence.py")),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "compression": {"method": "post_training_linear_symmetric", "dtype": "int4", "granularity": "per_block",
                        "block_size": 32, "weight_threshold": 2048, "operations": ["linear", "gather"],
                        "activations": "float32", "attention_weight_exceptions": None},
        "limitations": "Weight compression, not int4 activation arithmetic or QAT; saved-package assessment required.",
    }
    try:
        for path in (Path(__file__), Path(__file__).with_name("evidence.py"), ROOT / "uv.lock"):
            shutil.copyfile(path, output / path.name)
        run(args.source_package.expanduser().resolve(), args.source_assessment.expanduser().resolve(),
            args.baseline.expanduser().resolve(), output, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_report(output / "export-report.json", report)


if __name__ == "__main__":
    main()
