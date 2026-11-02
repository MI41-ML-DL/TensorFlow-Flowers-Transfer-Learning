"""Predict a new photograph with the validation-selected classical pipeline.

python -m src.predict --image path/to/flower.jpg
Input must be a flower from the five supported classes; no rejection model is
trained. Probability estimates are not calibrated guarantees of correctness.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from .common import ROOT, backbone, class_names, preprocessing, read_config, setup


def predict_image(path):
    config = read_config()
    setup(config)
    selected = json.loads((ROOT / "results/selected_benchmark.json").read_text())
    name, clf = selected["extractor"], selected["classifier"]
    saved = ROOT / f"models/{name}_imagenet.weights.h5"
    base = backbone(name, config, weights=None if saved.exists() else "imagenet")
    if saved.exists():
        base.load_weights(saved)
    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    image = tf.image.resize(tf.cast(image, tf.float32), [config["image_size"]]*2)
    feature = base(preprocessing(name)(image)[None], training=False).numpy()
    classifier = joblib.load(ROOT / f"models/{name}_{clf}.joblib")
    pred = int(classifier.predict(feature)[0])
    proba = classifier.predict_proba(feature)[0]
    names = class_names()
    try:
        shown = Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = Path(path).name  # do not print private absolute paths
    return {"image": shown, "extractor": name, "classifier": clf, "predicted_class": names[pred],
            "probabilities": {names[int(i)]: float(p) for i, p in zip(classifier.classes_, proba)},
            "decision_rule": "estimator.predict; SVM decision and probability argmax can differ"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image), indent=2, ensure_ascii=False))
