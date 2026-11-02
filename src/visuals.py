"""All figures are computed from saved real observations and predictions."""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image

from .common import CLASSIFIERS, EXTRACTORS, ROOT, class_names, manifest

FIGURES = ROOT / "results/figures"
COLORS = ["#d6a52f", "#4783b5", "#c86b85", "#8ba34a", "#9a60a0"]
MODEL_NAMES = {"mobilenetv2": "MobileNetV2", "resnet50": "ResNet50", "vgg16": "VGG16"}
CLF_NAMES = {"logreg": "Logistic regression", "svm": "SVM (RBF)", "randomforest": "Random Forest"}
VAL_COLOR, TEST_COLOR, ACCENT = "#70aaa3", "#3e6f91", "#b86c84"


def expected_figures(config):
    """Names of every PNG this module writes; used by the audit and the reports."""
    names = ["eda_class_counts", "eda_split_distribution", "eda_examples", "eda_dimensions",
             "benchmark_accuracy", "benchmark_macro_f1", "confusion_raw_grid", "confusion_normalized_grid",
             "typical_errors", "confident_errors", "common_error_pairs", "error_pairs_across_models",
             "feature_tsne", "freeze_depth_accuracy_trend", "freeze_learning_accuracy",
             "freeze_learning_loss", "freeze_best_confusion", "freeze_best_confusion_normalized"]
    for prefix in ("confusion_raw", "confusion_normalized"):
        names += [f"{prefix}_{e}_{c}" for e in EXTRACTORS for c in CLASSIFIERS]
    return names


def policy_name(last, total):
    return "all layers" if int(last) >= int(total) else f"last {int(last)}"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})


def photograph(row):
    """Use the original JPEG or its verified shipped demonstration copy."""
    from .common import sha256
    original = ROOT / row.path
    path = original if original.is_file() else ROOT / "data/demo_images" / f"{row.image_id}.jpg"
    if not path.is_file():
        raise FileNotFoundError(f"Photo {row.image_id} is unavailable. Run --stage prepare first.")
    if sha256(path) != row.sha256:
        raise ValueError(f"Photo checksum mismatch: {row.image_id}")
    return path


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def eda(df, names):
    counts = df.class_name.value_counts().reindex(names)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    bars = ax.bar(names, counts, color=COLORS, width=.65)
    ax.bar_label(bars, padding=4)
    ax.set(title=f"TensorFlow Flowers | all {len(df):,} official images", ylabel="Number of images",
           ylim=(0, counts.max() * 1.15))
    ax.yaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    save(fig, "eda_class_counts")

    counts = pd.crosstab(df.class_name, df.split).reindex(names)[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout="constrained")
    counts.plot.bar(ax=ax, color=["#3e6f91", "#69a6a0", "#d4a64d"], rot=0)
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=3)
    ax.set(title="One split for every model | duplicate groups kept together",
           xlabel="Flower class", ylabel="Number of images", ylim=(0, counts.to_numpy().max() * 1.15))
    ax.legend(title="Subset", frameon=False)
    save(fig, "eda_split_distribution")

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.8), layout="constrained")
    gallery_rows = []
    for j, cls in enumerate(names):
        examples = df[(df.class_name == cls) & (df.split == "train")].sample(2, random_state=42)
        for i, row in enumerate(examples.itertuples()):
            ax = axes[i, j]
            with Image.open(photograph(row)) as image:
                ax.imshow(image)
            ax.set_title(f"{cls}\n{row.image_id}", fontsize=10)
            ax.axis("off")
            gallery_rows.append(row.image_id)
    fig.suptitle("Training examples | two photographs per class", fontsize=16, weight="bold")
    save(fig, "eda_examples")
    pd.DataFrame({"image_id": gallery_rows}).to_csv(ROOT / "results/tables/eda_example_ids.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for i, cls in enumerate(names):
        rows = df[df.class_name == cls]
        axes[0].scatter(rows.width, rows.height, s=8, alpha=.30, color=COLORS[i], label=cls)
    axes[0].set(xlabel="Original width (pixels)", ylabel="Original height (pixels)", title="Original image dimensions")
    axes[0].legend(fontsize=8, markerscale=2, frameon=False)
    axes[1].hist(df.width / df.height, bins=35, color="#4783b5", edgecolor="white")
    axes[1].axvline(1, color="#c86b85", linestyle="--", label="Square")
    axes[1].set(xlabel="Width / height", ylabel="Number of images", title="Aspect ratio | common resize to 224 x 224")
    axes[1].legend(frameon=False)
    save(fig, "eda_dimensions")


def matrix_panel(ax, matrix, names, normalized=False, title="", vmax=None):
    values = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if normalized else matrix
    ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else vmax)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    threshold = .52 if normalized else (vmax or matrix.max()) * .52
    for i in range(len(names)):
        for j in range(len(names)):
            text = f"{values[i, j]:.1%}" if normalized else str(int(values[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] > threshold else "#17324a")


def benchmark_plots(names):
    metrics = pd.read_csv(ROOT / "results/tables/benchmark_metrics.csv")
    selected = json.loads((ROOT / "results/selected_benchmark.json").read_text(encoding="utf-8"))
    keys = [(e, c) for e in EXTRACTORS for c in CLASSIFIERS]
    labels = [f"{MODEL_NAMES[e]} + {CLF_NAMES[c]}" for e, c in keys]
    for metric, file, axis_label in (("accuracy", "benchmark_accuracy", "Accuracy"),
                                     ("macro_f1", "benchmark_macro_f1", "Macro-F1")):
        values = {split: np.array([metrics[(metrics.extractor == e) & (metrics.classifier == c) &
                                           (metrics.split == split)][metric].iloc[0] for e, c in keys])
                  for split in ("val", "test")}
        # Dot plot on a zoomed common axis: bars from zero would hide 1-3 point differences.
        low = min(v.min() for v in values.values())
        high = max(v.max() for v in values.values())
        fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
        y = np.arange(len(keys))[::-1]
        for i in range(len(keys)):
            ax.plot([values["val"][i], values["test"][i]], [y[i], y[i]], color="#c9d3d8", lw=2, zorder=1)
        ax.scatter(values["val"], y, s=70, color=VAL_COLOR, label="Validation (used for selection)", zorder=3)
        ax.scatter(values["test"], y, s=70, color=TEST_COLOR, marker="D", label="Held-out test", zorder=3)
        for i in range(len(keys)):
            fmt = (lambda v: f"{v:.1%}") if metric == "accuracy" else (lambda v: f"{v:.3f}")
            ax.annotate(fmt(values["test"][i]), (values["test"][i], y[i]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8, color=TEST_COLOR)
        ax.set_yticks(y, labels)
        for tick, (e, c) in zip(ax.get_yticklabels(), keys):
            if (e, c) == (selected["extractor"], selected["classifier"]):
                tick.set_fontweight("bold")
                tick.set_color(ACCENT)
        for k in range(1, len(EXTRACTORS)):
            ax.axhline(len(keys) - k * len(CLASSIFIERS) - .5, color="#e3e8ea", lw=1)
        ax.set_xlim(low - .015, high + .015)
        if metric == "accuracy":
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel(axis_label + " (zoomed axis)")
        ax.xaxis.grid(True, alpha=.2)
        ax.set_axisbelow(True)
        ax.set_title(f"{len(keys)} fixed candidates | highlighted = chosen on validation macro-F1")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        save(fig, file)

    matrices = {f"{e}_{c}": pd.read_csv(ROOT / f"results/tables/confusion_{e}_{c}.csv", index_col=0).to_numpy()
                for e, c in keys}
    vmax = max(m.max() for m in matrices.values())
    for normalized in (False, True):
        prefix = "confusion_normalized" if normalized else "confusion_raw"
        fig, axes = plt.subplots(len(EXTRACTORS), len(CLASSIFIERS), figsize=(14, 4.55 * len(EXTRACTORS)),
                                 layout="constrained", squeeze=False)
        for ax, (e, c) in zip(axes.ravel(), keys):
            row = metrics[(metrics.extractor == e) & (metrics.classifier == c) & (metrics.split == "test")].iloc[0]
            title = f"{MODEL_NAMES[e]} + {CLF_NAMES[c]}\ntest acc {row.accuracy:.1%} | macro-F1 {row.macro_f1:.3f}"
            matrix_panel(ax, matrices[f"{e}_{c}"], names, normalized, title, vmax)
            single, sax = plt.subplots(figsize=(6.2, 5.6), layout="constrained")
            matrix_panel(sax, matrices[f"{e}_{c}"], names, normalized, title, vmax)
            save(single, f"{prefix}_{e}_{c}")
        fig.suptitle("Held-out test | row-normalized proportions (diagonal = recall)" if normalized else
                     "Held-out test | raw counts (same color scale)", fontsize=17, weight="bold")
        save(fig, f"{prefix}_grid")


def error_gallery(rows, df, title, filename):
    rows = rows.head(12)
    nrows = max(1, int(np.ceil(len(rows)/4)))
    fig, axes = plt.subplots(nrows, 4, figsize=(13, nrows*3.05), squeeze=False, layout="constrained")
    lookup = df.set_index("image_id")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows.itertuples()):
        photo_row = lookup.loc[row.image_id].copy()
        photo_row["image_id"] = row.image_id
        path = photograph(photo_row)
        with Image.open(path) as image:
            ax.imshow(image)
        ax.set_title(f"True: {row.true_class}  |  Pred: {row.pred_class}\n"
                     f"p(pred)={row.confidence:.1%}  |  {row.image_id}", fontsize=9, color="#8a354f")
    if len(rows) == 0:
        axes[0, 0].text(.5, .5, "No errors in this test split", ha="center", va="center")
    fig.suptitle(title, fontsize=16, weight="bold")
    save(fig, filename)


def systematic_error_pairs(names):
    """Directed confusions counted for every classical pair and for the selected CNN."""
    records = []
    for e in EXTRACTORS:
        for c in CLASSIFIERS:
            m = pd.read_csv(ROOT / f"results/tables/confusion_{e}_{c}.csv", index_col=0)
            for t in names:
                for p in names:
                    if t != p:
                        records.append({"model": f"{e}_{c}", "true_class": t, "pred_class": p,
                                        "count": int(m.loc[t, p])})
    long = pd.DataFrame(records)
    table = long.pivot_table(index=["true_class", "pred_class"], columns="model", values="count").reset_index()
    models = [f"{e}_{c}" for e in EXTRACTORS for c in CLASSIFIERS]
    table["mean_classical"] = table[models].mean(axis=1)
    table["min_classical"] = table[models].min(axis=1)
    table["max_classical"] = table[models].max(axis=1)
    cnn = pd.read_csv(ROOT / "results/tables/confusion_freeze_best.csv", index_col=0)
    table["selected_cnn"] = [int(cnn.loc[t, p]) for t, p in zip(table.true_class, table.pred_class)]
    table = table.sort_values(["mean_classical", "true_class", "pred_class"], ascending=[False, True, True])
    table[["true_class", "pred_class", *models, "mean_classical", "min_classical", "max_classical",
           "selected_cnn"]].to_csv(ROOT / "results/tables/error_pairs_across_models.csv", index=False)
    return table, models


def error_plots(df, names):
    pairs = pd.read_csv(ROOT / "results/tables/common_error_pairs.csv")
    rows = pd.read_csv(ROOT / "results/tables/gallery_errors.csv")
    error_gallery(rows, df, "Typical errors | most frequent confusion pairs first", "typical_errors")
    confident = pd.read_csv(ROOT / "results/tables/typical_errors.csv")
    error_gallery(confident, df, "High-confidence errors | validation-selected classical model", "confident_errors")
    fig, ax = plt.subplots(figsize=(9.5, 5.4), layout="constrained")
    part = pairs.head(10).iloc[::-1]
    labels = [f"{r.true_class}  →  {r.pred_class}" for r in part.itertuples()]
    bars = ax.barh(labels, part["count"], color=ACCENT, height=.68)
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Number of test errors", title="Most frequent directed confusion pairs | selected classical model",
           xlim=(0, max(part["count"].max()*1.18, 1)))
    save(fig, "common_error_pairs")

    table, models = systematic_error_pairs(names)
    top = table.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    y = np.arange(len(top))
    for i, row in enumerate(top.itertuples()):
        values = [getattr(row, m) for m in models]
        ax.scatter(values, np.full(len(values), y[i]) + np.linspace(-.18, .18, len(values)),
                   s=14, color="#9aa9b2", alpha=.8, zorder=2)
        ax.plot([row.min_classical, row.max_classical], [y[i], y[i]], color="#d4dcdf", lw=6, zorder=1,
                solid_capstyle="round")
    ax.scatter(top.mean_classical, y, s=80, color=TEST_COLOR, marker="D", zorder=3,
               label=f"Mean of {len(models)} classical pairs")
    ax.scatter(top.selected_cnn, y, s=80, color=ACCENT, marker="o", zorder=4, label="Selected fine-tuned CNN")
    ax.set_yticks(y, [f"{r.true_class}  →  {r.pred_class}" for r in top.itertuples()])
    ax.set(xlabel="Test errors in this directed cell (grey dots = each classical pair)",
           title="Systematic confusions | the same cells recur across extractors and classifiers")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.grid(True, alpha=.2)
    ax.legend(handles=[Line2D([], [], color="#9aa9b2", marker="o", ls="", label=f"Each of {len(models)} classical pairs"),
                       Line2D([], [], color=TEST_COLOR, marker="D", ls="", markersize=8, label="Mean of classical pairs"),
                       Line2D([], [], color=ACCENT, marker="o", ls="", markersize=8, label="Selected fine-tuned CNN")],
              loc="lower right", frameon=False)
    save(fig, "error_pairs_across_models")


def feature_tsne(names, seed=42):
    """2-D t-SNE of test-set features: three frozen extractors and the selected fine-tuned CNN."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    selected = json.loads((ROOT / "results/selected_benchmark.json").read_text(encoding="utf-8"))
    errors = pd.read_csv(ROOT / f"results/predictions/{selected['extractor']}_{selected['classifier']}_test.csv")
    cnn_errors = pd.read_csv(ROOT / "results/predictions/freeze_best_test.csv")
    panels = [(e, f"Frozen {MODEL_NAMES[e]}", ROOT / f"results/features/features_{e}.npz", errors) for e in EXTRACTORS]
    tuned_path = ROOT / "results/features/features_finetuned_test.npz"
    if tuned_path.exists():
        with np.load(tuned_path) as data:
            last = int(data["unfreeze_last"])
        total = int(pd.read_csv(ROOT / "results/tables/freeze_depth_validation.csv").total_backbone_layers.iloc[0])
        panels.append(("finetuned", f"Fine-tuned MobileNetV2 ({policy_name(last, total)})", tuned_path, cnn_errors))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.0), layout="constrained")
    rows, summary = [], []
    for ax, (key, title, path, wrong_df) in zip(np.atleast_1d(axes), panels):
        with np.load(path) as data:
            x, y, ids = data["X_test"], data["y_test"], data["id_test"]
        z = PCA(n_components=50, random_state=seed).fit_transform(x)
        emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                   random_state=seed).fit_transform(z)
        # Descriptive cluster purity: share of each point's 10 nearest 2-D neighbours with its class.
        neigh = KNeighborsClassifier(n_neighbors=11).fit(emb, y).kneighbors(emb, return_distance=False)[:, 1:]
        purity = float(np.mean(y[neigh] == y[:, None]))
        for k, cls in enumerate(names):
            mask = y == k
            ax.scatter(emb[mask, 0], emb[mask, 1], s=9, alpha=.75, color=COLORS[k], label=cls)
        show_errors = key == selected["extractor"] or key == "finetuned"
        if show_errors:
            wrong = np.isin(ids, wrong_df.loc[~wrong_df.correct, "image_id"])
            ax.scatter(emb[wrong, 0], emb[wrong, 1], s=34, facecolors="none", edgecolors="black", linewidths=.8)
        ax.set_title(f"{title}\n{x.shape[1]} features | 10-NN purity {purity:.0%}", fontsize=11)
        ax.set(xticks=[], yticks=[])
        for spine in ("left", "bottom"):
            ax.spines[spine].set_alpha(.3)
        rows += [{"features": key, "image_id": i, "label": int(l), "tsne_1": float(a), "tsne_2": float(b)}
                 for i, l, (a, b) in zip(ids, y, emb)]
        summary.append({"features": key, "dimension": int(x.shape[1]), "knn10_purity_2d": purity})
    handles = [Line2D([], [], color=COLORS[k], marker="o", ls="", label=cls) for k, cls in enumerate(names)]
    handles.append(Line2D([], [], color="black", marker="o", markerfacecolor="none", ls="",
                          label="test error (selected classical pair / selected CNN)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(.5, -.07))
    fig.suptitle("t-SNE of test-set features | colored by true class; labels are not used by t-SNE",
                 fontsize=15, weight="bold")
    save(fig, "feature_tsne")
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/feature_tsne_test.csv", index=False)
    pd.DataFrame(summary).to_csv(ROOT / "results/tables/feature_tsne_purity.csv", index=False)


def freeze_plots(config, names):
    table = pd.read_csv(ROOT / "results/tables/freeze_depth_validation.csv").sort_values("frozen_layers")
    chosen = json.loads((ROOT / "results/selected_depth_test.json").read_text(encoding="utf-8"))
    best = chosen["best_unfreeze_last"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6), layout="constrained")
    ax.plot(table.frozen_layers, table.val_accuracy, "o-", color=TEST_COLOR, linewidth=2.4, markersize=8,
            label="Validation accuracy")
    ax.plot(table.frozen_layers, table.val_macro_f1, "s--", color=VAL_COLOR, linewidth=1.4, markersize=5,
            label="Validation macro-F1")
    pick = table[table.unfreeze_last == best].iloc[0]
    ax.scatter([pick.frozen_layers], [pick.val_accuracy], s=260, facecolors="none", edgecolors=ACCENT,
               linewidths=2, zorder=5,
               label=f"Selected on validation: {policy_name(best, pick.total_backbone_layers)} "
                     f"(test {chosen['test']['accuracy']:.1%})")
    acc = table.val_accuracy.to_numpy()
    for i, row in enumerate(table.itertuples()):
        neighbours = [acc[j] for j in (i - 1, i + 1) if 0 <= j < len(acc)]
        above = row.val_accuracy >= np.mean(neighbours)  # label away from the line
        ax.annotate(f"{policy_name(row.unfreeze_last, row.total_backbone_layers)}\n{row.val_accuracy:.1%}",
                    (row.frozen_layers, row.val_accuracy), xytext=(0, 12 if above else -32),
                    textcoords="offset points", ha="center", fontsize=9)
    low = min(table.val_accuracy.min(), table.val_macro_f1.min())
    high = max(table.val_accuracy.max(), table.val_macro_f1.max())
    ax.set(xticks=table.frozen_layers,
           xlabel=f"Frozen MobileNetV2 layers out of {int(table.total_backbone_layers.iloc[0])} "
                  f"(includes {int(table.frozen_batchnorm_layers.iloc[0])} always-frozen BatchNorm)",
           ylabel="Validation score, restored lowest-loss checkpoint",
           title="Accuracy vs depth of frozen layers | same start, LR, data order and epoch budget",
           ylim=(max(0, low - .025), min(1.005, high + .02)))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.grid(True, alpha=.2)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    save(fig, "freeze_depth_accuracy_trend")

    policies = list(config["unfreeze_last"])
    ncols = 3
    nrows = int(np.ceil(len(policies) / ncols))
    histories = {k: pd.read_csv(ROOT / f"results/tables/depth_{k}_history.csv") for k in policies}
    for field, title, filename in (("accuracy", "Accuracy", "freeze_learning_accuracy"),
                                    ("loss", "Cross-entropy loss", "freeze_learning_loss")):
        fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.3 * nrows), layout="constrained",
                                 sharex=True, sharey=True, squeeze=False)
        for ax in axes.ravel()[len(policies):]:
            ax.axis("off")
        for ax, last in zip(axes.ravel(), policies):
            h = histories[last]
            epochs = np.arange(1, len(h)+1)
            ax.plot(epochs, h[field], "o-", label="Train (dropout on)", color=TEST_COLOR, markersize=4)
            ax.plot(epochs, h[f"val_{field}"], "o-", label="Validation", color=ACCENT, markersize=4)
            row = table.loc[table.unfreeze_last == last].iloc[0]
            ax.axvline(int(row.restored_epoch), linestyle=":", color="#999999", label="Restored epoch")
            ax.set_title(f"Unfreeze {policy_name(last, row.total_backbone_layers)} | {int(row.frozen_layers)} frozen",
                         fontsize=10)
            ax.set_xticks(range(1, config["fine_tune_epochs"] + 1))
            ax.grid(alpha=.15)
            if field == "accuracy":
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        for ax in axes[-1]:
            ax.set_xlabel("Post-warmup epoch")
        for ax in axes[:, 0]:
            ax.set_ylabel(title)
        axes[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"Controlled fine-tuning | {title.lower()} (shared y-axis)", fontsize=16, weight="bold")
        save(fig, filename)
    matrix = pd.read_csv(ROOT / "results/tables/confusion_freeze_best.csv", index_col=0).to_numpy()
    for normal in (False, True):
        fig, ax = plt.subplots(figsize=(6.5, 5.8), layout="constrained")
        matrix_panel(ax, matrix, names, normal,
                     f"Selected MobileNetV2 | {policy_name(best, chosen['validation']['total_backbone_layers'])} | "
                     f"test acc {chosen['test']['accuracy']:.1%}",
                     matrix.max())
        save(fig, "freeze_best_confusion_normalized" if normal else "freeze_best_confusion")


def copy_demonstration_images(df):
    """Keep small, attributed real-image samples for offline presentation."""
    ids = set(pd.read_csv(ROOT / "results/tables/eda_example_ids.csv").image_id)
    for name in ("gallery_errors", "typical_errors"):
        ids.update(pd.read_csv(ROOT / f"results/tables/{name}.csv").head(12).image_id)
    ids.update(pd.read_csv(ROOT / "results/tables/duplicate_audit.csv").image_id)
    observations = ROOT / "results/tables/error_observations_ua.csv"
    if observations.exists():
        ids.update(pd.read_csv(observations).image_id)
    dest = ROOT / "data/demo_images"
    dest.mkdir(exist_ok=True)
    for row in df[df.image_id.isin(ids)].itertuples():
        source = ROOT / row.path
        if source.exists():
            shutil.copy2(source, dest / f"{row.image_id}.jpg")
    df[df.image_id.isin(ids)].to_csv(dest / "attribution_manifest.csv", index=False)
    license_path = ROOT / "data/flower_photos/LICENSE.txt"
    if license_path.exists():
        shutil.copy2(license_path, ROOT / "data/FLOWERS_LICENSE.txt")


def generate(config):
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    df, names = manifest(), class_names()
    eda(df, names)
    benchmark_plots(names)
    freeze_plots(config, names)
    error_plots(df, names)
    feature_tsne(names, config["seed"])
    copy_demonstration_images(df)
    missing = [n for n in expected_figures(config) if not (FIGURES / f"{n}.png").exists()]
    assert not missing, missing
    print("Figures:", len(expected_figures(config)), flush=True)
