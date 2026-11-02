"""Fit nine predeclared candidates (3 CNN x 3 classifiers), lock validation winner, then evaluate test."""
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import EXTRACTORS, ROOT, class_names, manifest, metric_record, save_json, sha256
from .features import load_features


def classifiers(config):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            C=1., max_iter=3000, solver="lbfgs", random_state=config["seed"])),
        "svm": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=1., gamma="scale", probability=True, random_state=config["seed"])),
        "randomforest": RandomForestClassifier(n_estimators=config["random_forest_trees"],
            max_features="sqrt", n_jobs=config["threads"], random_state=config["seed"]),
    }


def run(config):
    candidates, metrics = [], []
    names = class_names()
    for extractor in EXTRACTORS:
        data = load_features(config, extractor)
        for name, estimator in classifiers(config).items():
            key = f"{extractor}_{name}"
            print("Training", key, flush=True)
            t = perf_counter()
            estimator.fit(data["X_train"], data["y_train"])
            fit_seconds = perf_counter() - t
            prediction = estimator.predict(data["X_val"])
            score = metric_record(data["y_val"], prediction)
            row = {"extractor": extractor, "classifier": name, "split": "val", **score,
                   "fit_seconds": fit_seconds, "n": len(prediction)}
            metrics.append(row)
            candidates.append((extractor, name, estimator, score))
            joblib.dump(estimator, ROOT / f"models/{key}.joblib", compress=3)
            print(row, flush=True)
        data.close()
    # This file is written before the code below reads any test features/labels.
    winner = sorted(candidates, key=lambda x: (-x[3]["macro_f1"], -x[3]["accuracy"], x[0], x[1]))[0]
    selected = {"extractor": winner[0], "classifier": winner[1], "criterion": "validation macro-F1, then accuracy",
                "validation_metrics": winner[3], "manifest_sha256": sha256(ROOT / "data/manifest.csv"),
                "test_used_for_selection": False, "predeclared_candidates": len(candidates)}
    save_json(ROOT / "results/selected_benchmark.json", selected)
    pd.DataFrame(metrics).to_csv(ROOT / "results/tables/benchmark_validation.csv", index=False)
    chosen_predictions = None
    for extractor, name, estimator, _ in candidates:
        data = load_features(config, extractor)
        t = perf_counter()
        prediction = estimator.predict(data["X_test"])
        proba = estimator.predict_proba(data["X_test"])
        prediction_seconds = perf_counter() - t
        key = f"{extractor}_{name}"
        score = metric_record(data["y_test"], prediction)
        fit_seconds = next(r["fit_seconds"] for r in metrics if r["extractor"] == extractor and r["classifier"] == name)
        metrics.append({"extractor": extractor, "classifier": name, "split": "test", **score,
                        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                        "n": len(prediction)})
        # SVC.predict need not equal argmax(predict_proba); the actual decision is retained.
        columns = {int(label): i for i, label in enumerate(estimator.classes_)}
        confidence = np.asarray([proba[i, columns[int(p)]] for i, p in enumerate(prediction)])
        pred_df = pd.DataFrame({"image_id": data["id_test"], "true_label": data["y_test"],
                                "pred_label": prediction, "confidence": confidence})
        pred_df["true_class"] = pred_df.true_label.map(dict(enumerate(names)))
        pred_df["pred_class"] = pred_df.pred_label.map(dict(enumerate(names)))
        pred_df["correct"] = pred_df.true_label == pred_df.pred_label
        for i, cls in enumerate(names):
            pred_df[f"prob_{cls}"] = proba[:, columns[i]]
        pred_df.to_csv(ROOT / f"results/predictions/{key}_test.csv", index=False)
        matrix = confusion_matrix(data["y_test"], prediction, labels=range(len(names)))
        pd.DataFrame(matrix, index=names, columns=names).to_csv(ROOT / f"results/tables/confusion_{key}.csv")
        report = classification_report(data["y_test"], prediction, labels=list(range(len(names))),
                                       target_names=names, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(ROOT / f"results/tables/classification_report_{key}.csv")
        if (extractor, name) == (winner[0], winner[1]):
            chosen_predictions = pred_df
        data.close()
        print("TEST", key, score, flush=True)
    pd.DataFrame(metrics).to_csv(ROOT / "results/tables/benchmark_metrics.csv", index=False)
    errors = chosen_predictions[~chosen_predictions.correct].copy()
    errors = errors.sort_values(["confidence", "image_id"], ascending=[False, True])
    errors.to_csv(ROOT / "results/tables/typical_errors.csv", index=False)
    pairs = errors.groupby(["true_class", "pred_class"]).size().reset_index(name="count")
    pairs = pairs.sort_values(["count", "true_class", "pred_class"], ascending=[False, True, True])
    pairs.to_csv(ROOT / "results/tables/common_error_pairs.csv", index=False)
    # Gallery represents the most frequent pairs first, with up to four photos per pair.
    picks = []
    for pair in pairs.itertuples():
        picks.extend(errors[(errors.true_class == pair.true_class) &
                            (errors.pred_class == pair.pred_class)].head(4).image_id.tolist())
    errors.set_index("image_id").loc[picks[:12]].reset_index().to_csv(
        ROOT / "results/tables/gallery_errors.csv", index=False)
    return pd.DataFrame(metrics)
