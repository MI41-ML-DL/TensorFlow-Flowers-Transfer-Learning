"""Evidence checks: leakage, predictions, matrices and actual weight updates."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .common import (CLASSIFIERS, EXTRACTORS, ROOT, backbone, class_names, manifest, metric_record,
                     save_json, sha256)
from .visuals import expected_figures


def audit(config):
    df = manifest()
    names = class_names()
    results = []

    def check(name, condition, detail=""):
        results.append({"check": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    check("Official sample count", len(df) == 3670)
    check("One subset per duplicate group", df.groupby("sha256").split.nunique().max() == 1)
    check("Five classes in every split", all(df[df.split == s].label.nunique() == 5 for s in ("train", "val", "test")))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check(f"No shared content hashes: {a}/{b}",
              not (set(df[df.split == a].sha256) & set(df[df.split == b].sha256)))
    table = pd.read_csv(ROOT / "results/tables/benchmark_metrics.csv")
    n_pairs = len(EXTRACTORS) * len(CLASSIFIERS)
    check(f"{n_pairs} validation and {n_pairs} test records",
          len(table) == 2 * n_pairs and table.groupby("split").size().eq(n_pairs).all())
    for extractor in EXTRACTORS:
        with np.load(ROOT / f"results/features/features_{extractor}.npz") as data:
            for split in ("train", "val", "test"):
                rows = df[df.split == split]
                check(f"{extractor} {split} feature IDs", np.array_equal(data[f"id_{split}"], rows.image_id))
                check(f"{extractor} {split} feature labels", np.array_equal(data[f"y_{split}"], rows.label))
            for classifier in CLASSIFIERS:
                key = f"{extractor}_{classifier}"
                pred = pd.read_csv(ROOT / f"results/predictions/{key}_test.csv")
                matrix = pd.read_csv(ROOT / f"results/tables/confusion_{key}.csv", index_col=0).to_numpy()
                recomputed = confusion_matrix(pred.true_label, pred.pred_label, labels=range(5))
                check(f"{key}: matrix matches predictions", np.array_equal(matrix, recomputed) and matrix.sum() == len(pred))
                check(f"{key}: test order", np.array_equal(pred.image_id, data["id_test"]))
                model = joblib.load(ROOT / f"models/{key}.joblib")
                check(f"{key}: persisted estimator", np.array_equal(model.predict(data["X_test"]), pred.pred_label))
                score = metric_record(pred.true_label, pred.pred_label)
                saved = table[(table.extractor == extractor) & (table.classifier == classifier) & (table.split == "test")].iloc[0]
                check(f"{key}: metrics recompute", all(np.isclose(saved[k], v) for k, v in score.items()))
                if classifier != "randomforest":
                    check(f"{key}: scaler fitted only on train", model[0].n_samples_seen_ == len(data["y_train"]))
    selected = json.loads((ROOT / "results/selected_benchmark.json").read_text())
    ranked = table[table.split == "val"].sort_values(["macro_f1", "accuracy", "extractor", "classifier"],
                                                    ascending=[False, False, True, True]).iloc[0]
    check("Winner follows validation ranking", (selected["extractor"], selected["classifier"]) ==
          (ranked.extractor, ranked.classifier))
    freeze = pd.read_csv(ROOT / "results/tables/freeze_depth_validation.csv")
    check(f"{len(config['unfreeze_last'])} predeclared freeze policies", set(freeze.unfreeze_last) == set(config["unfreeze_last"]))
    check("Frozen plus trainable flags equals total",
          np.all(freeze.frozen_layers + freeze.trainable_layer_flags == freeze.total_backbone_layers))
    selected_depth = json.loads((ROOT / "results/selected_depth_test.json").read_text())
    best_val = freeze.sort_values(["val_accuracy", "val_loss", "unfreeze_last"], ascending=[False, True, True]).iloc[0]
    check("Selected CNN policy follows validation ranking",
          selected_depth["best_unfreeze_last"] == int(best_val.unfreeze_last))
    cnn_pred = pd.read_csv(ROOT / "results/predictions/freeze_best_test.csv")
    cnn_matrix = pd.read_csv(ROOT / "results/tables/confusion_freeze_best.csv", index_col=0).to_numpy()
    check("Selected CNN: matrix matches predictions",
          np.array_equal(cnn_matrix, confusion_matrix(cnn_pred.true_label, cnn_pred.pred_label, labels=range(5))))
    cnn_score = metric_record(cnn_pred.true_label, cnn_pred.pred_label)
    check("Selected CNN: test metrics recompute",
          all(np.isclose(selected_depth["test"][k], v) for k, v in cnn_score.items()))
    check("Selected CNN: test order", np.array_equal(cnn_pred.image_id, df[df.split == "test"].image_id))
    check("Every policy uses shared warmup and learning rate", freeze.shared_warmup_epochs.nunique() == 1 and
          freeze.fine_tune_learning_rate.nunique() == 1)
    # This verifies fine-tuning really happened and all frozen weights stayed fixed.
    import tensorflow as tf
    initial = backbone("mobilenetv2", config, weights=None)
    initial.load_weights(ROOT / "models/mobilenetv2_imagenet.weights.h5")
    initial_weights = {layer.name: layer.get_weights() for layer in initial.layers}
    for last in config["unfreeze_last"]:
        model = tf.keras.models.load_model(ROOT / f"models/depth_{last}.keras", compile=False)
        base = next(layer for layer in model.layers if layer.name.startswith("mobilenetv2"))
        frozen_delta, changed = 0., 0
        for layer in base.layers:
            for old, new in zip(initial_weights[layer.name], layer.get_weights()):
                delta = float(np.max(np.abs(old-new))) if old.size else 0.
                if layer.trainable:
                    changed += int(delta > 0)
                else:
                    frozen_delta = max(frozen_delta, delta)
        check(f"Depth {last}: frozen weights unchanged", frozen_delta == 0., f"max delta={frozen_delta}")
        check(f"Depth {last}: expected trainable updates", changed > 0 if last else changed == 0,
              f"changed trainable tensors={changed}")
        bn = [layer for layer in base.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
        check(f"Depth {last}: BatchNorm frozen", all(not layer.trainable for layer in bn))
        del model, base
    figures = expected_figures(config)
    missing = [name for name in figures if not (ROOT / f"results/figures/{name}.png").exists()]
    check(f"All {len(figures)} figures exist", not missing, ", ".join(missing))
    save_json(ROOT / "results/audit.json", {"status": "passed", "checks": results,
              "manifest_sha256": sha256(ROOT / "data/manifest.csv"),
              "note": "Metric and serialization checks reuse saved test predictions; no model selection is repeated."})
    pd.DataFrame(results).to_csv(ROOT / "results/tables/audit_checks.csv", index=False)
    print(f"AUDIT PASSED: {len(results)} evidence checks", flush=True)
    return results
