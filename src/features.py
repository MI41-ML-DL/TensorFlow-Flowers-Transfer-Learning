"""Frozen ImageNet inference with verified, ordered feature caches."""
import gc
import json
from time import perf_counter

import numpy as np
import pandas as pd

from .common import (ROOT, backbone, class_names, fingerprint, image_dataset, images_only,
                     manifest, save_json)


def extract(config, model_name):
    import tensorflow as tf
    df = manifest()
    path = ROOT / f"results/features/features_{model_name}.npz"
    meta_path = path.with_suffix(".json")
    expected = fingerprint(config, model_name)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["fingerprint"] != expected:
            raise ValueError(f"Stale {model_name} feature cache. Remove it explicitly before recomputing.")
        print("Using verified feature cache:", path.name, flush=True)
        return path
    tf.keras.backend.clear_session()
    network = backbone(model_name, config)
    network.trainable = False
    network.save_weights(ROOT / f"models/{model_name}_imagenet.weights.h5")
    arrays = {}
    started = perf_counter()
    for split in ("train", "val", "test"):
        rows = df[df.split == split]
        print(model_name, split, len(rows), flush=True)
        ds = image_dataset(rows, config, model_name)
        arrays[f"X_{split}"] = network.predict(ds.map(images_only), verbose=2).astype(np.float32)
        arrays[f"y_{split}"] = rows.label.to_numpy(dtype=np.int32)
        arrays[f"id_{split}"] = rows.image_id.to_numpy(dtype=str)
        assert len(arrays[f"X_{split}"]) == len(rows)
        assert np.isfinite(arrays[f"X_{split}"]).all()
    seconds = perf_counter() - started
    np.savez_compressed(path, **arrays)
    save_json(meta_path, {"fingerprint": expected, "class_names": class_names(),
                         "feature_dim": int(arrays["X_train"].shape[1]),
                         "extraction_seconds": seconds, "backbone_params": network.count_params()})
    print(f"{model_name}: {seconds:.1f} seconds", flush=True)
    del network, arrays
    tf.keras.backend.clear_session()
    gc.collect()
    return path


def load_features(config, name):
    meta = json.loads((ROOT / f"results/features/features_{name}.json").read_text(encoding="utf-8"))
    assert meta["fingerprint"] == fingerprint(config, name)
    arrays = np.load(ROOT / f"results/features/features_{name}.npz", allow_pickle=False)
    df = manifest()
    for split in ("train", "val", "test"):
        rows = df[df.split == split]
        assert np.array_equal(arrays[f"id_{split}"], rows.image_id.to_numpy())
        assert np.array_equal(arrays[f"y_{split}"], rows.label.to_numpy())
    return arrays
