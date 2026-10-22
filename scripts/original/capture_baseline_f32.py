"""Capture original-model float32 evidence for a fixed 512-token text variant."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/embeddinggemma-2"
RECEIPT = ROOT / "models/receipts/embeddinggemma-2.json"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LIMIT = 512
TOLERANCES = {
    "unit_norm_atol": 1e-5,
    "encode_atol": 1e-6,
    "encode_rtol": 1e-5,
    "padding_atol": 1e-5,
    "padding_rtol": 1e-4,
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model_source():
    source = json.loads(RECEIPT.read_text())
    if source["repository"] != "google/embeddinggemma-2" or source["revision"] != REVISION:
        raise ValueError("Model receipt does not identify the pinned official checkpoint")
    expected = {item["path"] for item in source["files"]}
    actual = {
        path.relative_to(MODEL).as_posix()
        for path in MODEL.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(MODEL).parts
    }
    if not expected or len(expected) != len(source["files"]) or expected != actual:
        raise ValueError("Local model file inventory differs from its receipt")
    for item in source["files"]:
        path = MODEL / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Local model hash or size mismatch: {item['path']}")
    return source


def processing_kwargs(padded=False):
    text = {"truncation": False, "padding": "max_length" if padded else False}
    if padded:
        text["max_length"] = LIMIT
    return {"text": text}


def prepare(model, text, prompt_name):
    if not text.strip():
        raise ValueError("Empty or whitespace-only content is rejected before prompting")
    features = model.preprocess(
        [text], prompt=model.prompts[prompt_name], processing_kwargs=processing_kwargs()
    )
    count = features["input_ids"].shape[1]
    if count > LIMIT:
        raise ValueError(f"Input has {count} prompted tokens; selected variant limit is {LIMIT}")
    if features["attention_mask"].shape != (1, count):
        raise ValueError("Expected one unpadded attention mask")
    if not torch.all(features["attention_mask"] == 1):
        raise ValueError("Unexpected padding in unpadded fixture")
    return features


def exact_length_text(model, target, prompt_name="Document"):
    # Construct real text and measure it through the official processor, including
    # the task prompt and special tokens. Never synthesize token IDs or truncate.
    for count in range(max(1, target - 64), target + 1):
        text = "planet " * count
        features = model.preprocess(
            [text], prompt=model.prompts[prompt_name], processing_kwargs=processing_kwargs()
        )
        if features["input_ids"].shape[1] == target:
            return text
    raise ValueError(f"Could not construct {target} prompted tokens for {prompt_name}")


def validate(embedding):
    if embedding.shape != (1, 768) or embedding.dtype != np.float32:
        raise ValueError(f"Expected float32 (1, 768), got {embedding.dtype} {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite embedding")
    np.testing.assert_allclose(
        np.linalg.norm(embedding, axis=1), 1.0,
        atol=TOLERANCES["unit_norm_atol"], rtol=0,
    )


def capture(model, features, prefix, arrays):
    extras = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            arrays[f"{prefix}__{key}"] = value.detach().cpu().numpy().copy()
        else:
            extras[key] = value
    # The official pipeline mutates the feature dict, so give it independent inputs.
    inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value
              for key, value in features.items()}
    embedding = model(inputs)["sentence_embedding"].detach().cpu().numpy().copy()
    validate(embedding)
    arrays[f"{prefix}__embedding"] = embedding
    return embedding, extras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output}")
    source = verify_model_source()
    provenance = {
        "model_source_sha256": sha256(RECEIPT),
        "script_sha256": sha256(Path(__file__)),
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
    }
    print(f"Verified checkpoint {source['revision']} ({len(source['files'])} files)", flush=True)
    model = SentenceTransformer(
        str(MODEL), device="cpu", local_files_only=True, trust_remote_code=False,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
    )
    model.eval()
    backbone = model[0].auto_model
    if backbone.config.model_type != "embedding_gemma2":
        raise ValueError("Expected EmbeddingGemma 2")
    if backbone.vision_tower is not None or backbone.audio_tower is not None:
        raise ValueError("Expected text-only model")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Expected CPU float32 parameters")
    if model[0].tokenizer.padding_side != "right" or not model[1].include_prompt:
        raise ValueError("Expected right padding and prompt-inclusive pooling")

    cases = [
        ("query_mars", "SearchQuery", "Which planet is known as the Red Planet?"),
        ("doc_mars", "Document", "Mars is known as the Red Planet."),
        ("doc_venus", "Document", "Venus has a thick atmosphere."),
        ("doc_saturn", "Document", "Saturn has prominent rings."),
        ("query_paraphrase", "SearchQuery", "What is the name of the red planet?"),
        ("doc_paraphrase", "Document", "The planet nicknamed the Red Planet is Mars."),
        ("doc_unrelated", "Document", "Bread dough rises when yeast produces carbon dioxide."),
        ("short", "Document", "Hi"),
        ("query_french", "SearchQuery", "Quelle planète est surnommée la planète rouge ?"),
        ("doc_arabic", "Document", "يُعرف المريخ بالكوكب الأحمر."),
        ("doc_japanese", "Document", "火星は赤い惑星として知られています。"),
    ]
    for length in (511, 512):
        cases.append((f"tokens_{length}", "Document", exact_length_text(model, length)))

    rejected = []
    for prompt_name in ("SearchQuery", "Document"):
        for name, text in (
            ("empty", ""), ("whitespace", " \t\n"),
            ("over_limit", exact_length_text(model, LIMIT + 1, prompt_name)),
        ):
            try:
                prepare(model, text, prompt_name)
            except ValueError as error:
                rejected.append({"id": name, "prompt_name": prompt_name, "reason": str(error)})
            else:
                raise ValueError(f"Failed to reject {name} for {prompt_name}")

    arrays, records, vectors = {}, [], {}
    with torch.inference_mode():
        for name, prompt_name, text in cases:
            features = prepare(model, text, prompt_name)
            count = features["input_ids"].shape[1]
            embedding, extras = capture(model, features, name, arrays)
            public = model.encode(
                [text], prompt_name=prompt_name, processing_kwargs=processing_kwargs(),
                show_progress_bar=False, convert_to_numpy=True,
            )
            np.testing.assert_allclose(
                embedding, public, atol=TOLERANCES["encode_atol"], rtol=TOLERANCES["encode_rtol"]
            )
            padded_features = model.preprocess(
                [text], prompt=model.prompts[prompt_name], processing_kwargs=processing_kwargs(True)
            )
            ids, mask = padded_features["input_ids"], padded_features["attention_mask"]
            if ids.shape != (1, LIMIT) or mask.shape != (1, LIMIT):
                raise ValueError(f"Expected fixed 512-token inputs for {name}")
            if not torch.equal(ids[:, :count], features["input_ids"]):
                raise ValueError(f"Padded token IDs differ for {name}")
            if not torch.all(mask[:, :count] == 1) or not torch.all(mask[:, count:] == 0):
                raise ValueError(f"Unexpected right-padding mask for {name}")
            if not torch.all(ids[:, count:] == model[0].tokenizer.pad_token_id):
                raise ValueError(f"Unexpected padding token IDs for {name}")
            padded, padded_extras = capture(model, padded_features, f"{name}__padded_512", arrays)
            np.testing.assert_allclose(
                padded, embedding, atol=TOLERANCES["padding_atol"], rtol=TOLERANCES["padding_rtol"]
            )
            vectors[name] = embedding[0]
            records.append({
                "id": name, "prompt_name": prompt_name, "text": text,
                "prompt": model.prompts[prompt_name], "prompted_text": model.prompts[prompt_name] + text,
                "token_count": count, "feature_extras": extras, "padded_feature_extras": padded_extras,
                "public_encode_max_abs_error": float(np.max(np.abs(embedding - public))),
                "padding_max_abs_error": float(np.max(np.abs(padded - embedding))),
            })
            print(f"Captured {name}: {count} tokens; unpadded and padded to 512", flush=True)

    documents = ["doc_mars", "doc_venus", "doc_saturn"]
    scores = np.stack([vectors[name] for name in documents]) @ vectors["query_mars"]
    ranking = [documents[index] for index in np.argsort(-scores)]
    if ranking[0] != "doc_mars":
        raise ValueError(f"Unexpected Mars retrieval ranking: {ranking}")
    # Detect source or environment changes during capture before publishing evidence.
    verify_model_source()
    for key, path in (("model_source_sha256", RECEIPT), ("script_sha256", Path(__file__)),
                      ("uv_lock_sha256", ROOT / "uv.lock")):
        if provenance[key] != sha256(path):
            raise ValueError(f"Provenance changed during capture: {path}")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "fixtures.npz", **arrays)
    with np.load(output / "fixtures.npz", allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    for src, name in ((RECEIPT, "model-source.json"), (Path(__file__), "capture_baseline_f32.py"),
                      (ROOT / "uv.lock", "uv.lock")):
        shutil.copyfile(src, output / name)
    metadata = {
        "stage": "original-model-baseline", "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": source["repository"], "revision": source["revision"], **provenance,
        "fixtures_sha256": sha256(output / "fixtures.npz"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "device": "cpu", "dtype": "float32", "attention": "eager", "text_only": True,
        "torch_num_threads": torch.get_num_threads(), "sequence_length": LIMIT, "batch_size": 1,
        "padding_side": model[0].tokenizer.padding_side,
        "policy": "Reject empty content and inputs above 512 prompted tokens; never truncate.",
        "pooling": {"mode": model[1].pooling_mode, "include_prompt": model[1].include_prompt},
        "validation_tolerances": TOLERANCES, "cases": records, "rejected": rejected,
        "retrieval": {"documents": documents, "scores": scores.tolist(), "ranking": ranking},
    }
    with (output / "metadata.json").open("x") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved {len(cases)} baseline cases and {len(rejected)} rejection checks to {output}")


if __name__ == "__main__":
    main()
