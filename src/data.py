"""Official TensorFlow Flowers archive, TFDS labels and a duplicate-safe split."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import tarfile
import urllib.request

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .common import ROOT, dirs, save_json, sha256

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ARCHIVE_HASH = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"


def prepare(config):
    import tensorflow_datasets as tfds
    dirs()
    # Metadata, including the nonalphabetical class order, comes from TFDS.
    names = tfds.builder("tf_flowers").info.features["label"].names
    image_root = ROOT / "data/flower_photos"
    if len(list(image_root.glob("*/*.jpg"))) != 3670:
        archive = ROOT / "data/downloads/flower_photos.tgz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists() or sha256(archive) != ARCHIVE_HASH:
            print("Downloading the official TensorFlow Flowers archive...", flush=True)
            urllib.request.urlretrieve(URL, archive)
        if sha256(archive) != ARCHIVE_HASH:
            raise ValueError("Dataset archive checksum mismatch")
        with tarfile.open(archive) as tar:
            # Reject traversal, links and unexpected roots (Python 3.11 compatible).
            for member in tar.getmembers():
                target = (ROOT / "data" / member.name).resolve()
                if not target.is_relative_to((ROOT / "data").resolve()) or member.issym() or member.islnk():
                    raise ValueError("Unsafe archive member")
            try:
                tar.extractall(ROOT / "data", filter="data")
            except TypeError:  # Python < 3.11.4: members were validated above
                tar.extractall(ROOT / "data")

    records = []
    for i, path in enumerate(sorted(image_root.glob("*/*.jpg"))):
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        records.append({"image_id": f"flower_{i:05d}", "path": path.relative_to(ROOT).as_posix(),
                        "label": names.index(path.parent.name), "class_name": path.parent.name,
                        "width": width, "height": height, "sha256": sha256(path)})
    df = pd.DataFrame(records)
    assert len(df) == 3670 and df.image_id.is_unique
    # Split unique file-content groups, then map back to all official records.
    # A conflicting-label pair remains together and retains its official labels.
    groups = df.drop_duplicates("sha256", keep="first")
    train, rest = train_test_split(groups, test_size=.30, random_state=config["seed"],
                                  stratify=groups.label)
    val, test = train_test_split(rest, test_size=.5, random_state=config["seed"],
                                stratify=rest.label)
    split_for_hash = {h: split for split, part in (("train", train), ("val", val), ("test", test))
                      for h in part.sha256}
    df["split"] = df.sha256.map(split_for_hash)
    assert df.groupby("sha256").split.nunique().max() == 1
    assert all(set(df[df.split == s].label) == set(range(5)) for s in ("train", "val", "test"))
    path = ROOT / "data/manifest.csv"
    new_csv = df.to_csv(index=False, lineterminator="\n")
    if path.exists() and path.read_text(encoding="utf-8") != new_csv:
        raise ValueError("Existing manifest differs. Use a separate project copy for a new split.")
    path.write_text(new_csv, encoding="utf-8")
    save_json(ROOT / "data/class_names.json", names)
    duplicates = df[df.duplicated("sha256", keep=False)].copy()
    duplicates["conflicting_labels"] = duplicates.groupby("sha256").label.transform("nunique") > 1
    duplicates.to_csv(ROOT / "results/tables/duplicate_audit.csv", index=False)
    counts = pd.crosstab(df.class_name, df.split).reindex(names)[["train", "val", "test"]]
    counts.to_csv(ROOT / "results/tables/split_counts.csv")
    save_json(ROOT / "results/data_quality.json", {
        "source": URL, "archive_sha256": ARCHIVE_HASH,
        "label_source": "tensorflow_datasets.builder('tf_flowers').info.features['label'].names",
        "raw_images": len(df), "unique_file_hashes": df.sha256.nunique(),
        "duplicate_groups": duplicates.sha256.nunique(),
        "conflicting_label_groups": duplicates[duplicates.conflicting_labels].sha256.nunique(),
        "duplicate_groups_crossing_splits": 0,
        "split_method": "stratified 70/15/15 on unique SHA-256 groups; all original records retained",
        "split_counts": {k: int(v) for k, v in df.split.value_counts().items()},
        "manifest_sha256": sha256(path), "seed": config["seed"],
        "warning": "Exact duplicates are grouped; near-duplicates are not exhaustively audited."})
    print(counts, flush=True)
    print("Manifest SHA-256:", sha256(path), flush=True)
    return df
