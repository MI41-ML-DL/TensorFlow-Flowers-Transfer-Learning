"""Shared configuration, fingerprints, runtime metadata and tf.data input."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

# Three ImageNet feature extractors; VGG16 is named explicitly on slide 14.
EXTRACTORS = ("mobilenetv2", "resnet50", "vgg16")
CLASSIFIERS = ("logreg", "svm", "randomforest")
DISPLAY_NAMES = {"mobilenetv2": "MobileNetV2", "resnet50": "ResNet50", "vgg16": "VGG16",
                 "logreg": "Logistic Regression", "svm": "SVM RBF", "randomforest": "Random Forest"}


def read_config(path=None):
    return json.loads(Path(path or ROOT / "configs/full.json").read_text(encoding="utf-8"))


def dirs():
    for name in ("data", "models", "report", "results/figures", "results/tables",
                 "results/predictions", "results/logs", "results/features", "results/cache"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup(config):
    import numpy as np
    import tensorflow as tf
    dirs()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    tf.keras.utils.set_random_seed(config["seed"])
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config["threads"])
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # A Jupyter kernel may already have initialized TensorFlow.
    tf.config.experimental.enable_op_determinism()
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)


def runtime_metadata(config):
    import tensorflow as tf
    packages = {}
    for name in ("tensorflow-cpu", "tensorflow", "keras", "tensorflow-datasets", "numpy", "pandas",
                 "scipy", "scikit-learn", "matplotlib", "seaborn", "Pillow", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "platform": platform.platform(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "gpu": [str(x) for x in tf.config.list_physical_devices("GPU")],
            "packages": packages, "config": config}


def manifest():
    import pandas as pd
    df = pd.read_csv(ROOT / "data/manifest.csv", dtype={"image_id": str})
    assert df.image_id.is_unique and df.path.is_unique
    assert set(df.split) == {"train", "val", "test"}
    assert df.groupby("sha256").split.nunique().max() == 1, "Duplicate leakage detected"
    return df


def class_names():
    return json.loads((ROOT / "data/class_names.json").read_text(encoding="utf-8"))


def fingerprint(config, model=None):
    import tensorflow as tf
    return {"manifest_sha256": sha256(ROOT / "data/manifest.csv"),
            "image_size": config["image_size"], "model": model,
            "weights": "imagenet", "preprocessing": "keras.applications." + str(model),
            "tensorflow": tf.__version__}


def images_only(images, labels):
    return images


def preprocessing(model):
    import tensorflow as tf
    if model == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if model == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input
    if model == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input
    raise ValueError(model)


def backbone(model, config, weights="imagenet"):
    import tensorflow as tf
    factory = {"mobilenetv2": tf.keras.applications.MobileNetV2,
               "resnet50": tf.keras.applications.ResNet50,
               "vgg16": tf.keras.applications.VGG16}[model]
    local_weights = ROOT / f"models/{model}_imagenet.weights.h5"
    use_local = weights == "imagenet" and local_weights.exists()
    network = factory(include_top=False, weights=None if use_local else weights, pooling="avg",
                      input_shape=(config["image_size"], config["image_size"], 3))
    if use_local:
        network.load_weights(local_weights)
    return network


def image_dataset(rows, config, model, training=False):
    """Stable image order for inference; shuffling is enabled only for train."""
    import numpy as np
    import tensorflow as tf
    paths = [str(ROOT / p) for p in rows.path]
    labels = rows.label.to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(rows), seed=config["seed"], reshuffle_each_iteration=True)
    prep = preprocessing(model)

    def load(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(tf.cast(image, tf.float32),
                               (config["image_size"], config["image_size"]), method="bilinear")
        return prep(image), label

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    return (ds.with_options(options).map(load, num_parallel_calls=2, deterministic=True)
            .batch(config["batch_size"]).prefetch(1))


def array_dataset(x, y, config, training=False):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(y), seed=config["seed"], reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.experimental_deterministic = True
    return ds.with_options(options).batch(config["batch_size"]).prefetch(1)


def metric_record(y, prediction):
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0))}
