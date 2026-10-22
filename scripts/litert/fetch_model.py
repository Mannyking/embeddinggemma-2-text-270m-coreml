"""Download the pinned CPU/GPU LiteRT-LM text bundle and generate its receipt."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "litert-community/embeddinggemma-2-text-270m-litert-lm"
REVISION = "9be6e8b90982095dc05c2bd162e4b954ee4dbac7"
FILENAME = "embeddinggemma-2-text-270m.litertlm"
DESTINATION = ROOT / "models/embeddinggemma-2-text-270m-litertlm"
RECEIPT = ROOT / "models/receipts/embeddinggemma-2-text-270m-litertlm.json"


def main():
    if DESTINATION.exists() or RECEIPT.exists():
        raise FileExistsError("Refusing to overwrite an existing model directory or receipt")
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Upstream revision differs from the pin")
    entries = {item.rfilename: item for item in info.siblings}
    selected = ["README.md", FILENAME]
    snapshot_download(REPOSITORY, revision=REVISION, allow_patterns=selected, local_dir=DESTINATION)
    records = []
    for name in selected:
        entry, path = entries[name], DESTINATION / name
        size = path.stat().st_size
        if size != entry.size:
            raise ValueError(f"File size mismatch: {name}")
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                sha256.update(block)
                git_blob.update(block)
        expected = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        actual = sha256.hexdigest() if entry.lfs is not None else git_blob.hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream content hash mismatch: {name}")
        records.append({"path": name, "bytes": size, "sha256": sha256.hexdigest(), "download_etag": expected})
        print(f"Verified {name}: {size:,} bytes", flush=True)
    receipt = {
        "repository": REPOSITORY, "revision": REVISION,
        "verification": "Local content matches pinned Hugging Face API metadata (LFS SHA-256 or Git blob SHA-1); metadata is not independently authenticated.",
        "files": records,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPT.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(f"Receipt: {RECEIPT}")


if __name__ == "__main__":
    main()
