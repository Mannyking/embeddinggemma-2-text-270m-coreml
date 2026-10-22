"""Download and hash-verify the pinned NanoNQ corpus, questions, and relevance labels."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "zeta-alpha-ai/NanoNQ"
REVISION = "77540146379abf95df8326a3c5bb9eb21c7146c3"
DATASET = ROOT / "datasets/nanonq"
FILES = ["README.md", "corpus/train-00000-of-00001.parquet",
         "queries/train-00000-of-00001.parquet", "qrels/train-00000-of-00001.parquet"]


def main():
    if DATASET.exists():
        raise FileExistsError(f"Refusing to overwrite the dataset directory: {DATASET}")
    info = HfApi().dataset_info(REPOSITORY, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Dataset revision differs from the pin")
    entries = {item.rfilename: item for item in info.siblings}
    snapshot_download(REPOSITORY, repo_type="dataset", revision=REVISION,
                      allow_patterns=FILES, local_dir=DATASET)
    records = []
    for name in FILES:
        path, entry = DATASET / name, entries[name]
        size = path.stat().st_size
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                sha.update(block)
                blob.update(block)
        expected = entry.lfs.sha256 if entry.lfs is not None else entry.blob_id
        if size != entry.size or (sha.hexdigest() if entry.lfs is not None else blob.hexdigest()) != expected:
            raise ValueError(f"Upstream size/hash mismatch: {name}")
        records.append({"path": name, "bytes": size, "sha256": sha.hexdigest(), "download_etag": expected})
    receipt = {"repository": REPOSITORY, "revision": REVISION, "files": records,
               "verification": "Local files match upstream LFS SHA-256 or Git blob SHA-1 metadata; metadata is not independently authenticated."}
    with (DATASET / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(f"Verified {len(records)} files at {REVISION}; dataset: {DATASET}")


if __name__ == "__main__":
    main()
