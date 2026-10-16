"""Check local text-only PyTorch inference; this is not Core ML parity evidence."""

import argparse
import os
from pathlib import Path

# Set before importing Hugging Face libraries. This script never downloads weights.
os.environ["HF_HUB_OFFLINE"] = "1"

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models/embeddinggemma-2"
QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Mars is known as the Red Planet.",
    "Venus has a thick atmosphere.",
    "Saturn has prominent rings.",
]
LABELS = ["Mars", "Venus", "Saturn"]
PROCESSING_KWARGS = {"text": {"truncation": False}}


def encode_checked(model, texts, prompt_name):
    """Check prompted token lengths without allowing implicit truncation."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Inputs must contain non-empty text")
    prompt = model.prompts[prompt_name]
    features = model.preprocess(
        texts, prompt=prompt, processing_kwargs=PROCESSING_KWARGS
    )
    lengths = features["attention_mask"].sum(dim=1).tolist()
    limit = model[0].max_seq_length
    if limit is None or limit <= 0:
        raise ValueError("The local model must declare a positive input length limit")
    if any(length > limit for length in lengths):
        raise ValueError(f"Prompted token lengths {lengths} exceed model limit {limit}")
    print(f"{prompt_name}: {prompt!r}; token lengths: {lengths}")
    return model.encode(
        texts,
        prompt_name=prompt_name,
        processing_kwargs=PROCESSING_KWARGS,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        parser.error(f"Download the official checkpoint to {model_path} first")

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(model_path),
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
    )
    model.eval()
    backbone = model[0].auto_model
    if backbone.config.model_type != "embedding_gemma2":
        raise ValueError("Expected the official EmbeddingGemma 2 architecture")
    if backbone.vision_tower is not None or backbone.audio_tower is not None:
        raise ValueError("Vision and audio encoders must be disabled")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Expected float32 model parameters")

    print(f"Model: {model_path}\nCPU float32, eager attention, text only")
    print(f"Padding side: {model[0].tokenizer.padding_side}")
    with torch.inference_mode():
        query = encode_checked(model, [QUERY], "SearchQuery")[0]
        docs = encode_checked(model, DOCUMENTS, "Document")

    vectors = np.vstack([query, docs])
    if vectors.shape != (4, 768) or vectors.dtype != np.float32:
        raise ValueError(f"Expected float32 (4, 768), got {vectors.dtype} {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"Embeddings are not unit normalized: {norms}")

    scores = docs @ query
    print(f"\nQuery: {QUERY}\nUnit norms [query, Mars, Venus, Saturn]: {norms}")
    print("Retrieval ranking (cosine similarity):")
    for rank, index in enumerate(np.argsort(-scores), start=1):
        print(f"  {rank}. {LABELS[index]:6} {scores[index]:+.6f}")
    if int(np.argmax(scores)) != 0:
        raise ValueError("Expected Mars to rank first")
    print("Original-model smoke test passed; Core ML has not been assessed.")


if __name__ == "__main__":
    main()
