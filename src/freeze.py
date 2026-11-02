"""Controlled freeze-depth experiment with optional exact frozen-prefix caching.

Caching changes only where the immutable forward pass is computed. It does not
freeze additional weights or truncate the gradients in the trainable suffix.
No augmentation is used: a cached prefix would be inappropriate for changing
input images. All BatchNorm layers remain frozen and are called in inference.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

from .common import (ROOT, backbone, class_names, fingerprint, image_dataset, images_only,
                     array_dataset, manifest, metric_record, save_json)
from .features import load_features


def make_head(config):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1280,)),
        tf.keras.layers.Dropout(config["dropout"], seed=config["seed"]),
        tf.keras.layers.Dense(5, activation="softmax", name="flower_probabilities"),
    ], name="flower_head")


def compile_model(model, lr):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)


def warmup(config):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    data = load_features(config, "mobilenetv2")
    head = make_head(config)
    compile_model(head, config["warmup_learning_rate"])
    history = head.fit(array_dataset(data["X_train"], data["y_train"], config, True),
                       validation_data=array_dataset(data["X_val"], data["y_val"], config),
                       epochs=config["warmup_epochs"], verbose=2, shuffle=False)
    head.save_weights(ROOT / "models/shared_warmup_head.weights.h5")
    pd.DataFrame(history.history).assign(epoch=lambda d: np.arange(1, len(d)+1)).to_csv(
        ROOT / "results/tables/warmup_history.csv", index=False)
    data.close()
    return history.history


def configure(base, last):
    import tensorflow as tf
    base.trainable = True
    boundary = len(base.layers) - last if last else len(base.layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = (last > 0 and i >= boundary and
                           not isinstance(layer, tf.keras.layers.BatchNormalization))
    bn = [l for l in base.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    assert all(not l.trainable for l in bn)
    flags = sum(int(l.trainable) for l in base.layers)
    return {"unfreeze_last": int(last), "total_backbone_layers": len(base.layers),
            "frozen_layers": len(base.layers)-flags, "trainable_layer_flags": flags,
            "weighted_trainable_layers": sum(bool(l.trainable_weights) for l in base.layers),
            "trainable_backbone_params": sum(int(np.prod(w.shape)) for w in base.trainable_weights),
            "frozen_batchnorm_layers": len(bn)}


def frozen_frontier(base, boundary):
    """All edges crossing the cut, including shortcut edges in residual blocks."""
    import tensorflow as tf
    positions = {id(layer): i for i, layer in enumerate(base.layers)}
    frontier = []
    for layer in base.layers[boundary:]:
        for tensor in tf.nest.flatten(layer.input):
            source = tensor._keras_history.operation
            if positions[id(source)] < boundary and all(tensor is not t for t in frontier):
                frontier.append(tensor)
    if not frontier:
        raise ValueError("No frozen-prefix frontier")
    inputs = {tensor.name: tensor for tensor in frontier}
    prefix = tf.keras.Model(base.input, inputs, name="frozen_prefix")
    suffix = tf.keras.Model(inputs, base.output, name="trainable_suffix")
    return prefix, suffix


def cache_prefix(prefix, config, last):
    """Write float32 activations batchwise; no quantization or massive RAM copy."""
    df = manifest()
    rows = df[df.split != "test"].reset_index(drop=True)
    folder = ROOT / f"results/cache/prefix_{last}"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {}
    shapes = {key: tuple(int(x) for x in tensor.shape[1:]) for key, tensor in prefix.output.items()}
    for i, (key, shape) in enumerate(shapes.items()):
        arrays[key] = np.lib.format.open_memmap(folder / f"activation_{i}.npy", mode="w+",
                                               dtype="float32", shape=(len(rows), *shape))
    offset = 0
    started = perf_counter()
    for images, _ in image_dataset(rows, config, "mobilenetv2"):
        values = prefix(images, training=False)
        size = len(images)
        for key, value in values.items():
            arrays[key][offset:offset+size] = value.numpy()
        offset += size
    for array in arrays.values():
        array.flush()
    assert offset == len(rows)
    seconds = perf_counter()-started
    save_json(folder / "metadata.json", {"fingerprint": fingerprint(config, "mobilenetv2"),
              "unfreeze_last": last, "shapes": {k: list(v) for k, v in shapes.items()},
              "dtype": "float32", "rows": len(rows), "seconds": seconds})
    return arrays, rows, shapes, seconds


def memmap_dataset(arrays, rows, split, shapes, config, training=False):
    import tensorflow as tf
    positions = np.flatnonzero(rows.split.to_numpy() == split)
    labels = rows.label.to_numpy(dtype=np.int32)
    epoch = [0]

    def batches():
        indices = positions.copy()
        if training:
            np.random.default_rng(config["seed"] + epoch[0]).shuffle(indices)
            epoch[0] += 1
        for start in range(0, len(indices), config["batch_size"]):
            idx = indices[start:start+config["batch_size"]]
            yield {key: np.asarray(value[idx]) for key, value in arrays.items()}, labels[idx]

    signature = ({key: tf.TensorSpec((None, *shape), tf.float32) for key, shape in shapes.items()},
                 tf.TensorSpec((None,), tf.int32))
    ds = tf.data.Dataset.from_generator(batches, output_signature=signature)
    ds = ds.apply(tf.data.experimental.assert_cardinality(
        int(np.ceil(len(positions) / config["batch_size"]))))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.with_options(options).prefetch(1)


def run_policy(config, last):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    base = backbone("mobilenetv2", config)
    info = configure(base, last)
    head = make_head(config)
    head.load_weights(ROOT / "models/shared_warmup_head.weights.h5")
    inputs = tf.keras.Input((config["image_size"], config["image_size"], 3), name="preprocessed_image")
    full_model = tf.keras.Model(inputs, head(base(inputs, training=False)), name=f"flowers_depth_{last}")
    prefix_seconds = 0.
    arrays = None
    started = perf_counter()
    if last == 0:
        data = load_features(config, "mobilenetv2")
        rows = manifest().query("split != 'test'").reset_index(drop=True)
        ids = np.concatenate([data["id_train"], data["id_val"]])
        x = np.concatenate([data["X_train"], data["X_val"]])
        position = {image_id: i for i, image_id in enumerate(ids)}
        arrays = {"features": x[[position[i] for i in rows.image_id]]}
        shapes = {"features": (1280,)}
        cached_inputs = {"features": tf.keras.Input((1280,), name="features")}
        fit_model = tf.keras.Model(cached_inputs, head(cached_inputs["features"]))
        train_ds = memmap_dataset(arrays, rows, "train", shapes, config, True)
        val_ds = memmap_dataset(arrays, rows, "val", shapes, config)
        y_val = rows[rows.split == "val"].label.to_numpy(dtype=np.int32)
    elif config["cache_frozen_prefix"]:
        # last == all layers: the frozen "prefix" is the preprocessed input image itself.
        prefix, suffix = frozen_frontier(base, max(1, len(base.layers)-last))
        cached_inputs = {key: tf.keras.Input(shape=tensor.shape[1:], name=key)
                         for key, tensor in prefix.output.items()}
        fit_model = tf.keras.Model(cached_inputs, head(suffix(cached_inputs, training=False)))
        # An explicit equivalence check catches omitted residual shortcut edges.
        first_images, _ = next(iter(image_dataset(manifest().query("split == 'val'").head(4),
                                                  config, "mobilenetv2")))
        direct = full_model(first_images, training=False).numpy()
        cached = fit_model(prefix(first_images, training=False), training=False).numpy()
        max_difference = float(np.max(np.abs(direct-cached)))
        np.testing.assert_allclose(direct, cached, rtol=1e-5, atol=1e-6)
        info["prefix_equivalence_max_abs_error"] = max_difference
        arrays, rows, shapes, prefix_seconds = cache_prefix(prefix, config, last)
        train_ds = memmap_dataset(arrays, rows, "train", shapes, config, True)
        val_ds = memmap_dataset(arrays, rows, "val", shapes, config)
        y_val = rows[rows.split == "val"].label.to_numpy(dtype=np.int32)
    else:
        df = manifest()
        train_ds = image_dataset(df[df.split == "train"], config, "mobilenetv2", True)
        val_ds = image_dataset(df[df.split == "val"], config, "mobilenetv2")
        y_val = df[df.split == "val"].label.to_numpy(dtype=np.int32)
        fit_model = full_model
    compile_model(fit_model, config["fine_tune_learning_rate"])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                  patience=config["early_stopping_patience"], restore_best_weights=True),
                 tf.keras.callbacks.CSVLogger(ROOT / f"results/tables/depth_{last}_history.csv")]
    history = fit_model.fit(train_ds, validation_data=val_ds, epochs=config["fine_tune_epochs"],
                            callbacks=callbacks, verbose=2, shuffle=False)
    val_loss, val_accuracy = fit_model.evaluate(val_ds, verbose=0)
    val_probabilities = fit_model.predict(val_ds.map(images_only), verbose=0)
    val_pred = val_probabilities.argmax(axis=1)
    assert np.isclose(np.mean(val_pred == y_val), val_accuracy, atol=1e-6)
    full_model.save(ROOT / f"models/depth_{last}.keras")
    # Verify cache and full network still agree after the trainable weights changed.
    val_rows = manifest().query("split == 'val'").head(4)
    images, _ = next(iter(image_dataset(val_rows, config, "mobilenetv2")))
    post_fit = full_model.predict(images, verbose=0)
    np.testing.assert_allclose(post_fit, val_probabilities[:4], rtol=2e-4, atol=2e-5)
    info.update({"val_accuracy": float(val_accuracy), "val_loss": float(val_loss),
                 "val_macro_f1": metric_record(y_val, val_pred)["macro_f1"],
                 "epochs_run": len(history.history["loss"]),
                 "restored_epoch": int(np.argmin(history.history["val_loss"]))+1,
                 "prefix_cache_seconds": prefix_seconds, "total_seconds": perf_counter()-started,
                 "shared_warmup_epochs": config["warmup_epochs"],
                 "fine_tune_learning_rate": config["fine_tune_learning_rate"],
                 "prefix_cached": bool(config["cache_frozen_prefix"] and last > 0)})
    save_json(ROOT / f"results/depth_{last}_validation.json", info)
    print("FREEZE RESULT", info, flush=True)
    if last == 0:
        data.close()
    del full_model, fit_model, head, base, train_ds, val_ds
    if arrays is not None:
        del arrays
        if last > 0 and not config["keep_prefix_cache"]:
            shutil.rmtree(ROOT / f"results/cache/prefix_{last}")
    tf.keras.backend.clear_session()
    gc.collect()
    return info


def finalize(config):
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix, classification_report
    infos = [json.loads((ROOT / f"results/depth_{k}_validation.json").read_text(encoding="utf-8"))
             for k in config["unfreeze_last"]]
    table = pd.DataFrame(infos)
    table.to_csv(ROOT / "results/tables/freeze_depth_validation.csv", index=False)
    best = sorted(infos, key=lambda x: (-x["val_accuracy"], x["val_loss"], x["unfreeze_last"]))[0]
    lock = {"criterion": "restored-checkpoint validation accuracy, then lower val loss",
            "best_unfreeze_last": best["unfreeze_last"], "validation": best,
            "test_used_for_selection": False}
    save_json(ROOT / "results/selected_depth_validation.json", lock)
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(ROOT / f"models/depth_{best['unfreeze_last']}.keras", compile=False)
    rows = manifest().query("split == 'test'")
    proba = model.predict(image_dataset(rows, config, "mobilenetv2").map(images_only), verbose=2)
    pred = proba.argmax(axis=1)
    y = rows.label.to_numpy(dtype=np.int32)
    metrics = metric_record(y, pred)
    metrics["test_loss"] = float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-7, 1)).mean())
    save_json(ROOT / "results/selected_depth_test.json", {**lock, "test": metrics, "n_test": len(y)})
    names = class_names()
    pd.DataFrame(confusion_matrix(y, pred, labels=range(5)), index=names, columns=names).to_csv(
        ROOT / "results/tables/confusion_freeze_best.csv")
    pd.DataFrame(classification_report(y, pred, target_names=names, labels=list(range(5)),
                 zero_division=0, output_dict=True)).T.to_csv(ROOT / "results/tables/freeze_best_classification_report.csv")
    predictions = pd.DataFrame({"image_id": rows.image_id, "true_label": y, "pred_label": pred,
                                 "confidence": proba.max(axis=1), "correct": y == pred})
    predictions.to_csv(ROOT / "results/predictions/freeze_best_test.csv", index=False)
    # Penultimate (GAP) features of the fine-tuned backbone, used only for the t-SNE illustration.
    tuned = next(layer for layer in model.layers if layer.name.startswith("mobilenetv2"))
    features = tuned.predict(image_dataset(rows, config, "mobilenetv2").map(images_only), verbose=0)
    np.savez_compressed(ROOT / "results/features/features_finetuned_test.npz",
                        X_test=features.astype(np.float32), y_test=y, id_test=rows.image_id.to_numpy(dtype=str),
                        unfreeze_last=np.int32(best["unfreeze_last"]))
    print("SELECTED FREEZE TEST", metrics, flush=True)
    return table
