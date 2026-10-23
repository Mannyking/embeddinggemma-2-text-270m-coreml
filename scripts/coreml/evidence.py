"""Hash and fixture checks shared by Core ML export and saved-package assessment."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts/baseline-f32-512"
RECEIPT = ROOT / "models/receipts/embeddinggemma-2.json"
MODEL = ROOT / "models/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LENGTH = 512
# Keep the predecessor's F32 acceptance profile explicit and fixed before assessment.
F32_PARITY = {"atol": 1e-4, "rtol": 1e-3, "minimum_cosine": 0.9999, "unit_norm_atol": 1e-4}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_hashes(directory):
    return {path.relative_to(directory).as_posix(): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def environment():
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("coremltools", "torch", "transformers", "sentence-transformers", "numpy")},
    }


def write_report(path, report):
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_baseline(directory):
    """Load saved evidence without invoking the original model or tokenizing again."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata["stage"] != "original-model-baseline" or metadata["revision"] != REVISION
            or metadata["sequence_length"] != LENGTH or metadata["padding_side"] != "right"
            or metadata["dtype"] != "float32" or not metadata["text_only"]):
        raise ValueError("Expected the pinned text-only float32 512-token baseline")
    for filename, field in (
        ("fixtures.npz", "fixtures_sha256"), ("model-source.json", "model_source_sha256"),
        ("capture_baseline_f32.py", "script_sha256"), ("uv.lock", "uv_lock_sha256"),
    ):
        if sha256(directory / filename) != metadata[field]:
            raise ValueError(f"Baseline hash mismatch: {filename}")
    source = json.loads((directory / "model-source.json").read_text())
    if source["repository"] != "google/embeddinggemma-2" or source["revision"] != REVISION:
        raise ValueError("Baseline source snapshot does not identify the pinned model")
    names = [case["id"] for case in metadata["cases"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected unique baseline fixture IDs")
    source_config = next(item for item in source["files"] if item["path"] == "config.json")
    # Token IDs are checked against the pinned model's vocabulary and pad ID.
    # These are identity-bound constants, not a request to load original weights.
    if not source_config["sha256"]:
        raise ValueError("Missing pinned configuration hash")
    ids, masks, expected, unpadded = [], [], [], []
    with np.load(directory / "fixtures.npz", allow_pickle=False) as data:
        for case in metadata["cases"]:
            name, count = case["id"], case["token_count"]
            if not 0 < count <= LENGTH:
                raise ValueError(f"Invalid prompted length for {name}")
            prefix = f"{name}__padded_512"
            token_ids, mask = data[f"{prefix}__input_ids"], data[f"{prefix}__attention_mask"]
            if token_ids.shape != (1, LENGTH) or mask.shape != (1, LENGTH):
                raise ValueError(f"Expected (1, 512) input tensors: {name}")
            if token_ids.dtype.kind not in "iu" or mask.dtype.kind not in "iu":
                raise ValueError(f"Expected integer IDs and masks: {name}")
            if np.any(token_ids < 0) or np.any(token_ids >= 262144):
                raise ValueError(f"Invalid token IDs: {name}")
            if not np.all(mask[:, :count] == 1) or not np.all(mask[:, count:] == 0):
                raise ValueError(f"Expected nonempty right padding: {name}")
            if not np.all(token_ids[:, count:] == 0):
                raise ValueError(f"Unexpected pad IDs: {name}")
            if not np.array_equal(token_ids[:, :count], data[f"{name}__input_ids"]):
                raise ValueError(f"Padded and unpadded token IDs differ: {name}")
            if not np.array_equal(mask[:, :count], data[f"{name}__attention_mask"]):
                raise ValueError(f"Padded and unpadded attention masks differ: {name}")
            padded_embedding = data[f"{prefix}__embedding"]
            original_embedding = data[f"{name}__embedding"]
            for value in (padded_embedding, original_embedding):
                if value.dtype != np.float32 or value.shape != (1, 768) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid expected embedding: {name}")
                np.testing.assert_allclose(np.linalg.norm(value, axis=1), 1, atol=1e-5, rtol=0)
            np.testing.assert_allclose(padded_embedding, original_embedding, atol=1e-5, rtol=1e-4)
            ids.append(token_ids.astype(np.int32))
            masks.append(mask.astype(np.int32))
            expected.append(padded_embedding.copy())
            unpadded.append(original_embedding.copy())
    links = {
        "baseline_directory": str(directory),
        "baseline_metadata_sha256": sha256(directory / "metadata.json"),
        "baseline_fixtures_sha256": metadata["fixtures_sha256"],
        "model_source_sha256": metadata["model_source_sha256"],
        "baseline_script_sha256": metadata["script_sha256"],
        "baseline_uv_lock_sha256": metadata["uv_lock_sha256"],
    }
    return metadata, names, list(zip(ids, masks)), np.concatenate(expected), np.concatenate(unpadded), links


def verify_local_model(expected_receipt_hash):
    if sha256(RECEIPT) != expected_receipt_hash:
        raise ValueError("Current model receipt differs from the saved baseline")
    source = json.loads(RECEIPT.read_text())
    expected = {item["path"] for item in source["files"]}
    actual = {p.relative_to(MODEL).as_posix() for p in MODEL.rglob("*")
              if p.is_file() and ".cache" not in p.relative_to(MODEL).parts}
    if actual != expected or len(expected) != len(source["files"]):
        raise ValueError("Local model inventory differs from its receipt")
    for item in source["files"]:
        path = MODEL / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Local model size/hash mismatch: {item['path']}")


def compare(actual, expected):
    if actual.shape != (1, 768) or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError("Expected finite float32 (1, 768) output")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    norm = float(np.linalg.norm(a))
    cosine = float(np.sum(a * b) / (norm * np.linalg.norm(b))) if norm else 0.0
    metrics = {"max_absolute_error": float(np.max(np.abs(a - b))),
               "cosine": cosine, "norm": norm}
    passed = (np.allclose(a, b, atol=F32_PARITY["atol"], rtol=F32_PARITY["rtol"])
              and abs(norm - 1) <= F32_PARITY["unit_norm_atol"]
              and cosine >= F32_PARITY["minimum_cosine"])
    return metrics, bool(passed)
