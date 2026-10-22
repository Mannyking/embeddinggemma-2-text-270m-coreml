"""Export a checked fixed-shape float32 Core ML text embedder; assess it separately."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"

import coremltools as ct
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

from evidence import (BASELINE, LENGTH, MODEL, REVISION, ROOT, environment, file_hashes,
                      load_baseline, sha256, verify_local_model, write_report)


class Embedder(torch.nn.Module):
    """Keep the official backbone and head; provide traceable tensor attention masks."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model[0].auto_model
        self.head = torch.nn.ModuleList(list(model.children())[1:])
        config = self.backbone.language_model.config
        if any(layer.self_attn.is_causal for layer in self.backbone.language_model.layers):
            raise ValueError("Expected bidirectional attention on every layer")
        positions = torch.arange(LENGTH)
        self.register_buffer("local_allowed", (
            (positions[:, None] - positions[None, :]).abs() <= config.sliding_window
        )[None, None])

    def attention_masks(self, attention_mask):
        allowed = attention_mask[:, None, None, :].to(torch.bool).expand(1, 1, LENGTH, LENGTH)
        blocked = torch.finfo(torch.float32).min
        return {
            "full_attention": torch.where(allowed, 0.0, blocked),
            "sliding_attention": torch.where(allowed & self.local_allowed, 0.0, blocked),
        }

    def forward(self, input_ids, attention_mask):
        # The composite model maps media placeholder IDs to PAD before text
        # embedding, even with no media inputs. Express the same rule with
        # where rather than its boolean __or__, unsupported by Core ML Tools 9.
        safe_ids = input_ids
        for token_id in (self.backbone.config.image_token_id, self.backbone.config.video_token_id,
                         self.backbone.config.audio_token_id):
            safe_ids = torch.where(input_ids == token_id, self.backbone.config.text_config.pad_token_id, safe_ids)
        tokens = self.backbone.language_model(
            input_ids=safe_ids, attention_mask=self.attention_masks(attention_mask)
        )[0]
        features = {"token_embeddings": tokens, "attention_mask": attention_mask}
        for module in self.head:
            features = module(features)
        return features["sentence_embedding"]


def run(output, baseline, report):
    metadata, names, arrays, expected, _, links = load_baseline(baseline)
    report.update(links)
    report["fixture_order"] = names
    report["revision"] = REVISION
    verify_local_model(links["model_source_sha256"])
    inputs = [(torch.from_numpy(ids), torch.from_numpy(mask)) for ids, mask in arrays]
    report["stage"] = "load_original_model"
    model = SentenceTransformer(
        str(MODEL), device="cpu", local_files_only=True, trust_remote_code=False,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
    ).eval()
    backbone = model[0].auto_model
    if backbone.config.model_type != "embedding_gemma2" or backbone.vision_tower is not None or backbone.audio_tower is not None:
        raise ValueError("Expected the text-only EmbeddingGemma 2 backbone")
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Expected CPU float32 model parameters")
    if model[0].tokenizer.padding_side != "right" or not model[1].include_prompt:
        raise ValueError("Expected right padding and prompt-inclusive pooling")
    for case in metadata["cases"]:
        if model.prompts[case["prompt_name"]] != case["prompt"]:
            raise ValueError(f"Prompt differs from baseline: {case['id']}")
    wrapper = Embedder(model).eval()
    config = backbone.language_model.config
    report["sliding_window"] = config.sliding_window
    report["attention_mask_value"] = torch.finfo(torch.float32).min
    report["wrapper_checks"] = []
    with torch.no_grad():
        report["stage"] = "check_masks_and_wrapper"
        for index, (ids, mask) in enumerate(inputs):
            mask_kwargs = dict(config=config, inputs_embeds=torch.zeros(1, LENGTH, config.hidden_size), attention_mask=mask)
            original_masks = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }
            for key, value in wrapper.attention_masks(mask).items():
                # With no padding and a window spanning the whole sequence,
                # upstream may elide the mask. None is exactly zero additive bias.
                original_mask = original_masks[key]
                if original_mask is None:
                    original_mask = torch.zeros_like(value)
                torch.testing.assert_close(value, original_mask, atol=0, rtol=0)
            # Re-run the unmodified official pipeline on the exact saved fixed inputs.
            original = model({"input_ids": ids, "attention_mask": mask,
                              **metadata["cases"][index]["padded_feature_extras"]})["sentence_embedding"].numpy()
            actual = wrapper(ids, mask).numpy()
            for value in (original, actual):
                np.testing.assert_allclose(value, expected[index:index+1], atol=1e-5, rtol=1e-4)
            report["wrapper_checks"].append({
                "id": names[index], "mask_equality": True,
                "upstream_elided_masks": [key for key, value in original_masks.items() if value is None],
                "original_max_absolute_error": float(np.max(np.abs(original - expected[index:index+1]))),
                "wrapper_max_absolute_error": float(np.max(np.abs(actual - expected[index:index+1]))),
            })
            print(f"Checked original pipeline, masks, wrapper: {names[index]}", flush=True)
        report["stage"] = "trace"
        traced = torch.jit.trace(wrapper, inputs[0], check_trace=False)
        report["stage"] = "check_trace"
        report["trace_checks"] = []
        for index, args in enumerate(inputs):
            actual = traced(*args).numpy()
            np.testing.assert_allclose(actual, expected[index:index+1], atol=1e-5, rtol=1e-4)
            report["trace_checks"].append({"id": names[index], "max_absolute_error": float(np.max(np.abs(actual - expected[index:index+1])))})
    report["stage"] = "convert"
    print("Converting checked trace to float32 ML Program", flush=True)
    converted = ct.convert(
        traced, source="pytorch", convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=(1, LENGTH), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, LENGTH), dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
        skip_model_load=True,
    )
    converted.short_description = "EmbeddingGemma 2 text-only float32, fixed 512 prompted tokens"
    converted.input_description["input_ids"] = "Token IDs with official task prompt and special tokens; right pad with 0"
    converted.input_description["attention_mask"] = "1 for valid tokens, 0 for right padding; reject empty or over-limit text before tokenizing"
    converted.output_description["embedding"] = "One 768-dimensional L2-normalized text embedding"
    package = output / "EmbeddingGemma2F32.mlpackage"
    converted.save(str(package))
    report["package_name"] = package.name
    report["package_hashes"] = file_hashes(package)
    report["package_size_bytes"] = sum(p.stat().st_size for p in package.rglob("*") if p.is_file())
    report["specification_version"] = converted.get_spec().specificationVersion
    verify_local_model(links["model_source_sha256"])
    if load_baseline(baseline)[-1] != links:
        raise ValueError("Baseline changed during export")
    if report["uv_lock_sha256"] != sha256(ROOT / "uv.lock"):
        raise ValueError("Environment lock changed during export")
    report["stage"] = "export_complete"
    report["status"] = "exported"
    report["artifact_parity"] = "Unassessed; run assess_f32.py on the saved package"
    print(f"Export saved: {package}; saved-package assessment is still required", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh export directory")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "stage": "verify_inputs",
        **environment(), "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "script_sha256": sha256(Path(__file__)), "evidence_script_sha256": sha256(Path(__file__).with_name("evidence.py")),
        "precision": "float32", "input_dtype": "int32", "input_shape": [1, LENGTH],
        "output_dtype": "float32", "output_shape": [1, 768], "minimum_deployment_target": "iOS17/macOS14",
        "wrapper_tolerances": {"atol": 1e-5, "rtol": 1e-4},
        "conversion_note": "Core ML Tools 9.0 does not list Torch 2.14 as tested; saved-package assessment is required.",
    }
    try:
        for src in (Path(__file__), Path(__file__).with_name("evidence.py"), ROOT / "uv.lock"):
            shutil.copyfile(src, output / src.name)
        run(output, args.baseline.expanduser().resolve(), report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        write_report(output / "export-report.json", report)


if __name__ == "__main__":
    main()
