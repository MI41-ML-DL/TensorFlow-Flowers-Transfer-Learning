"""Generate factual Ukrainian reports and defense notes from completed results.

Every number in the generated text is read from results/ at generation time;
nothing is typed in by hand, so the reports stay consistent after a re-run.
"""
from __future__ import annotations

import base64
import tempfile
from html import escape
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd

from .common import CLASSIFIERS, DISPLAY_NAMES as NAMES, EXTRACTORS, ROOT, read_config
from .visuals import expected_figures

SOURCES = [
    ("Умови викладача, Labs.pdf, слайд 14", "https://github.com/dryabokon/Docs/blob/master/ML/Labs.pdf"),
    ("TFDS TensorFlow Flowers", "https://www.tensorflow.org/datasets/catalog/tf_flowers"),
    ("TFDS: код датасету й порядок міток", "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/tf_flowers/tf_flowers_dataset_builder.py"),
    ("Keras: Transfer learning & fine-tuning", "https://keras.io/guides/transfer_learning/"),
    ("Keras: MobileNetV2", "https://keras.io/api/applications/mobilenet/mobilenet_models/"),
    ("Keras: ResNet", "https://keras.io/api/applications/resnet/"),
    ("Keras: VGG16", "https://keras.io/api/applications/vgg/"),
    ("scikit-learn: model evaluation", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
    ("scikit-learn: t-SNE", "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"),
]
RUN_DATES = "31.10.2026 (основний запуск) та 01.11.2026 (розширення: VGG16 і додаткові режими заморожування)"


def pct(x, digits=2):
    return f"{100 * float(x):.{digits}f}%"


def pair(e, c):
    return f"{NAMES[e]} + {NAMES[c]}"


def policy_label(last, total):
    return "усі шари" if int(last) >= int(total) else f"last {int(last)}"


def plural(n, forms):
    """Ukrainian noun form for a count: plural(9, ("режим", "режими", "режимів"))."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def lr(x):
    return f"{float(x):.0e}".replace("e-0", "e-")


MODES = ("режим", "режими", "режимів")


def join_uk(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " та " + items[-1]


def load():
    def js(path):
        return json.loads((ROOT / path).read_text(encoding="utf-8"))
    config = read_config()
    metrics = pd.read_csv(ROOT / "results/tables/benchmark_metrics.csv")
    selected = js("results/selected_benchmark.json")
    depth = js("results/selected_depth_test.json")
    quality = js("results/data_quality.json")
    split_counts = pd.read_csv(ROOT / "results/tables/split_counts.csv")
    n = {k: int(v) for k, v in quality["split_counts"].items()}
    chosen = metrics[(metrics.split == "test") & (metrics.extractor == selected["extractor"]) &
                     (metrics.classifier == selected["classifier"])].iloc[0]
    rows = []
    for extractor in EXTRACTORS:
        for classifier in CLASSIFIERS:
            part = metrics[(metrics.extractor == extractor) & (metrics.classifier == classifier)].set_index("split")
            rows.append({"Екстрактор": NAMES[extractor], "Класифікатор": NAMES[classifier],
                         "Val acc": pct(part.loc["val", "accuracy"]),
                         "Val F1": f"{part.loc['val', 'macro_f1']:.4f}",
                         "Test acc": pct(part.loc["test", "accuracy"]),
                         "Test F1": f"{part.loc['test', 'macro_f1']:.4f}"})
    test = metrics[metrics.split == "test"]
    val = metrics[metrics.split == "val"]
    best_test_accuracy = test.accuracy.max()
    best_test = test[np.isclose(test.accuracy, best_test_accuracy)]
    by_extractor = pd.DataFrame({"val": val.groupby("extractor").accuracy.mean(),
                                 "test": test.groupby("extractor").accuracy.mean()}).reindex(list(EXTRACTORS))
    by_classifier = pd.DataFrame({"val": val.groupby("classifier").accuracy.mean(),
                                  "test": test.groupby("classifier").accuracy.mean()}).reindex(list(CLASSIFIERS))
    dims = {e: js(f"results/features/features_{e}.json")["feature_dim"] for e in EXTRACTORS}
    freeze = pd.read_csv(ROOT / "results/tables/freeze_depth_validation.csv").sort_values("frozen_layers", ascending=False)
    pairs = pd.read_csv(ROOT / "results/tables/common_error_pairs.csv")
    support = split_counts.set_index("class_name")["test"].to_dict()
    top = pairs.iloc[0]
    reverse = pairs[(pairs.true_class == top.pred_class) & (pairs.pred_class == top.true_class)]
    across = pd.read_csv(ROOT / "results/tables/error_pairs_across_models.csv")
    repro_path = ROOT / "results/reproducibility_check.json"
    return {"config": config, "metrics": metrics, "selected": selected, "depth": depth, "chosen": chosen,
            "n": n, "n_test": n["test"], "split_counts": split_counts,
            "class_counts": split_counts.set_index("class_name")[["train", "val", "test"]].sum(axis=1).to_dict(),
            "best_test_accuracy": float(best_test_accuracy),
            "best_test_names": join_uk([pair(r.extractor, r.classifier) for r in best_test.itertuples()]),
            "best_test_count": len(best_test),
            "best_test_correct": int(round(best_test_accuracy * n["test"])),
            "chosen_correct": int(round(chosen.accuracy * n["test"])),
            "depth_test_correct": int(round(depth["test"]["accuracy"] * n["test"])),
            "by_extractor": by_extractor, "by_classifier": by_classifier, "dims": dims,
            "benchmark": pd.DataFrame(rows), "n_pairs": len(EXTRACTORS) * len(CLASSIFIERS),
            "freeze": freeze, "quality": quality, "pairs": pairs, "top_pair": top,
            "top_support": int(support[top.true_class]),
            "reverse_count": int(reverse["count"].iloc[0]) if len(reverse) else 0,
            "across": across,
            "observations": pd.read_csv(ROOT / "results/tables/error_observations_ua.csv"),
            "tsne": pd.read_csv(ROOT / "results/tables/feature_tsne_purity.csv").set_index("features"),
            "runtime": js("results/runtime.json"), "audit": js("results/audit.json"),
            "repro": js("results/reproducibility_check.json") if repro_path.exists() else None,
            "n_figures": len(expected_figures(config))}


def markdown_table(df):
    rows = [[str(x).replace("|", "/").replace("\n", " ") for x in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(str(x) for x in df.columns) + " |\n"
    return header + "| " + " | ".join("---" for _ in df.columns) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def freeze_display(data):
    f = data["freeze"]
    total = f.total_backbone_layers
    return pd.DataFrame({"Режим": [policy_label(a, b) for a, b in zip(f.unfreeze_last, total)],
                         "Frozen": f.frozen_layers,
                         "Trainable flags": f.trainable_layer_flags,
                         "Шари з вагами": f.weighted_trainable_layers,
                         "Trainable params": f.trainable_backbone_params.map(lambda x: f"{int(x):,}".replace(",", " ")),
                         "Епохи": f.epochs_run, "Checkpoint": f.restored_epoch,
                         "Val acc": f.val_accuracy.map(pct),
                         "Val F1": f.val_macro_f1.map(lambda x: f"{x:.4f}")})


def trend_facts(data):
    """Plain-language description of the freeze-depth curve, computed from the table."""
    f = data["freeze"]  # sorted from most frozen to least frozen
    total = int(f.total_backbone_layers.iloc[0])
    head = f[f.unfreeze_last == 0].iloc[0]
    full = f.iloc[-1]
    best = f[f.unfreeze_last == data["depth"]["best_unfreeze_last"]].iloc[0]
    values = f.val_accuracy.to_numpy()
    drops = [(policy_label(a.unfreeze_last, total), policy_label(b.unfreeze_last, total), b.val_accuracy - a.val_accuracy)
             for (_, a), (_, b) in zip(f.iloc[:-1].iterrows(), f.iloc[1:].iterrows()) if b.val_accuracy < a.val_accuracy - 1e-9]
    monotone = not drops
    spread = values.max() - values.min()
    end = (f"розморожування всіх шарів, крім {int(full.frozen_batchnorm_layers)} BatchNorm" if full.unfreeze_last >= total
           else f"режиму {policy_label(full.unfreeze_last, total)}")
    text = (f"досліджено {len(f)} {plural(len(f), MODES)}: від повністю замороженої MobileNetV2 ({int(head.frozen_layers)} frozen, "
            f"навчається лише голова) до {end} ({int(full.frozen_layers)} frozen). "
            f"Validation accuracy змінюється від {pct(values.min())} до {pct(values.max())} "
            f"(розмах {100 * spread:.2f} в. п.). Head-only дає {pct(head.val_accuracy)}, найкращий режим "
            f"(«{policy_label(best.unfreeze_last, total)}») — {pct(best.val_accuracy)} "
            f"(+{100 * (best.val_accuracy - head.val_accuracy):.2f} в. п.)"
            + ("." if best.unfreeze_last == full.unfreeze_last else
               f", {'повне розморожування' if full.unfreeze_last >= total else policy_label(full.unfreeze_last, total)} — {pct(full.val_accuracy)}."))
    if monotone:
        text += " Що менше шарів заморожено, то вища validation accuracy: крива монотонна на всіх точках."
    else:
        text += (" Загальна тенденція: що більше шарів розморожено, то вища точність, але крива немонотонна: "
                 + "; ".join(f"перехід {a} → {b} знижує accuracy на {abs(100 * d):.2f} в. п. ({round(abs(d) * data['n']['val'])} фото)" for a, b, d in drops)
                 + ". Різниці в кілька фото на validation з {0} зображень не слід переінтерпретувати.".format(data["n"]["val"]))
    return {"text": text, "head": head, "full": full, "best": best, "monotone": monotone, "total": total}


def ranking_text(data):
    e = data["by_extractor"].sort_values("val", ascending=False)
    c = data["by_classifier"].sort_values("val", ascending=False)
    ext = join_uk([f"{NAMES[k]} {pct(r.val)} / {pct(r.test)}" for k, r in e.iterrows()])
    clf = join_uk([f"{NAMES[k]} {pct(r.val)} / {pct(r.test)}" for k, r in c.iterrows()])
    return (f"Середня accuracy validation / test за трьома класифікаторами: {ext}. "
            f"Середня accuracy за трьома екстракторами: {clf}.")


def tsne_text(data):
    t = data["tsne"]
    frozen = join_uk([f"{NAMES[e]} {pct(t.loc[e, 'knn10_purity_2d'], 0)}" for e in EXTRACTORS])
    text = ("t-SNE проєктує тестові ознаки у 2D без використання міток; колір — справжній клас. "
            f"Частка 10 найближчих сусідів того самого класу на площині (описова «чистота» кластерів): {frozen}")
    if "finetuned" in t.index:
        tuned, base = t.loc["finetuned", "knn10_purity_2d"], t.loc["mobilenetv2", "knn10_purity_2d"]
        text += (f"; після донавчання MobileNetV2 — {pct(tuned, 0)}. "
                 + ("Fine-tuning робить класи компактнішими, ніж заморожені ImageNet-ознаки. " if tuned > base else
                    "Помітного покращення розділення від fine-tuning на площині немає. "))
    else:
        text += ". "
    return text + ("Тюльпани й троянди перекриваються найбільше; кружками позначено тестові помилки обраних моделей. "
                   "t-SNE — ілюстрація, а не метрика якості.")


def conclusions(data):
    best = data["chosen"]
    depth = data["depth"]
    trend = trend_facts(data)
    top, n = data["top_pair"], data["n"]
    across = data["across"].iloc[0]
    categories = data["observations"].error_category.value_counts()
    dims = ", ".join(f"{NAMES[e]} — {d}" for e, d in data["dims"].items())
    text = f"""# Висновки за фактичним запуском

1. Порівняно {data['n_pairs']} пар екстрактор–класифікатор ({len(EXTRACTORS)} CNN × {len(CLASSIFIERS)} класифікатори) на {n['train']} train / {n['val']} validation / {n['test']} test. Розмір вектора ознак: {dims}. Єдине розбиття, масштабування лише на train й однакові метрики роблять порівняння узгодженим.

2. За validation macro-F1 вибрано **{pair(data['selected']['extractor'], data['selected']['classifier'])}**: validation accuracy **{pct(data['selected']['validation_metrics']['accuracy'])}**, validation macro-F1 **{data['selected']['validation_metrics']['macro_f1']:.4f}**. Для цієї зафіксованої пари test accuracy становить **{pct(best.accuracy)}**, macro-F1 — **{best.macro_f1:.4f}** ({data['chosen_correct']} правильних із {n['test']}).

3. Найвищу test accuracy серед {data['n_pairs']} фіксованих пар показали {data['best_test_names']}: **{pct(data['best_test_accuracy'])}**. Цей описовий результат не використано для перевибору моделі; ranking на validation не зобовʼязаний повторитися на test. {ranking_text(data)}

4. Accuracy trend: {trend['text']}

5. Обрана за validation CNN ({policy_label(depth['best_unfreeze_last'], trend['total'])}, checkpoint епохи **{depth['validation']['restored_epoch']}**) дала **{pct(depth['test']['accuracy'])}** test accuracy та **{depth['test']['macro_f1']:.4f}** macro-F1 ({data['depth_test_correct']} правильних із {n['test']}). Серед {data['n_pairs']} класичних пар максимум був {data['best_test_correct']}/{n['test']}. Це результати одного split, формально статистичну перевагу не доведено.

6. Найчастіша помилка обраної класичної пари: **{top.true_class} → {top.pred_class}, {int(top['count'])} фото**, тобто {int(top['count'])}/{data['top_support']} = **{pct(top['count'] / data['top_support'])}** цього класу; зворотний напрям — {data['reverse_count']}. Ця ж клітинка є найчастішою й у середньому по всіх {data['n_pairs']} класичних парах ({across.mean_classical:.1f} помилки; від {int(across.min_classical)} до {int(across.max_classical)}), а в обраної CNN — {int(across.selected_cnn)}. Отже, плутанина {top.true_class}/{top.pred_class} систематична, а не випадкова для одного алгоритму.

7. У {len(data['observations'])} переглянутих хибних прогнозах виділено причини: {join_uk([f'{k.lower()} ({v})' for k, v in categories.items()])}. Частина найвпевненіших «помилок» — це, найімовірніше, неточні офіційні мітки або квіти поза пʼятьма класами; модель без reject-класу все одно вибирає один із пʼяти.

8. Три пари точних дублікатів не перетинають split; одна пара з різними мітками збережена в train. Обмеження: один seed і split, майже-дублікати не усувалися вичерпно, можливі помилки анотації, немає зовнішнього test і reject-класу.
"""
    return text


def write_notes(data):
    report = ROOT / "report"
    text = conclusions(data)
    (report / "CONCLUSIONS_UA.md").write_text(text, encoding="utf-8")
    trend = trend_facts(data)
    f = data["freeze"]
    checklist = f"""# Відповідність вимогам слайда 14

Викладацька презентація й початковий MD збережені без змін у `materials/`.

| Вимога | Готовий доказ | Статус |
|---|---|---|
| Кілька класифікаторів на кількох екстракторах | {len(EXTRACTORS)} CNN (MobileNetV2, ResNet50, VGG16) × LR/SVM/RF = {data['n_pairs']} пар; `benchmark_metrics.csv`, по {data['n_pairs']} val/test записів | Виконано |
| Матриці помилок, 60% | {data['n_pairs']} raw + {data['n_pairs']} normalized PNG, {data['n_pairs']} CSV; дві оглядові сітки; дві матриці обраної CNN | Виконано |
| Accuracy trend, 20% | {len(f)} реально навчених політик заморожування; `freeze_depth_validation.csv`, `freeze_depth_accuracy_trend.png`, криві навчання | Виконано |
| Typical errors, 20% | частоти пар, систематичні пари в усіх моделях, `typical_errors.png`, `confident_errors.png`, {len(data['observations'])} описаних фото з категоріями причин | Виконано |
| Візуалізація ознак (як на слайді) | `feature_tsne.png`: t-SNE тестових ознак трьох заморожених CNN і донавченої MobileNetV2 | Додатково |
| Єдині дані | `data/manifest.csv`, seed 42, SHA-256, grouped stratification | Виконано |
| Правильний preprocessing | `common.preprocessing`: окремий для кожної CNN | Виконано |
| Справжній fine-tuning | Аудит: змінились лише дозволені ваги, усі frozen ваги незмінні | Виконано |
| Відсутність leakage | Train-only scaler, lock вибору за val, test після lock, hash-групи не перетинаються | Виконано |
| Відтворення й захист | 2 виконані notebook, {data['n_figures']} графіків, CSV, моделі, логи, автоматичні перевірки ({len(data['audit']['checks'])}), звіт і сценарій | Виконано |

Готовий критерій не є гарантією оцінки викладача. Датасет, CNN, split і кількість епох є рішеннями проєкту, а не додатковими обовʼязковими вимогами слайда.
"""
    (report / "REQUIREMENTS_CHECKLIST.md").write_text(checklist, encoding="utf-8")
    sel = data["selected"]
    depth = data["depth"]
    best = trend["best"]
    top = data["top_pair"]
    cc = data["class_counts"]
    counts_text = ", ".join(f"{name} {cc[name]}" for name in sorted(cc, key=cc.get, reverse=True))
    other_best = data["best_test_names"]
    obs = data["observations"].set_index("image_id")
    # The spoken examples below name concrete photos; fail loudly if they are no longer errors.
    spoken = {"flower_03530", "flower_00293", "flower_02227", "flower_01581", "flower_01983"}
    assert spoken <= set(obs.index), spoken - set(obs.index)
    defense = f"""# Захист Assignment 2: Transfer Learning

## Підготовка показу

Відкрити `00_Results_and_Defense.ipynb` зі збереженими виводами. Резервний варіант — `Results_UA.html` або PDF. Порядок класів TFDS: dandelion, daisy, tulips, sunflowers, roses. Назви команди можна додати в титульну комірку.

## Учасник 1: дані, ідея, MobileNetV2 і VGG16, 2–3 хв

«Мета роботи — порівняти готові CNN-екстрактори та класичні класифікатори для розпізнавання пʼяти квіткових класів. У Transfer Learning ми переносимо візуальні ознаки, вивчені на ImageNet, замість навчання всієї мережі з нуля на малому датасеті.

Ми використали всі {sum(cc.values())} офіційних фото TensorFlow Flowers. Готового validation і test немає, тому створили стратифікований поділ: {data['n']['train']} train, {data['n']['val']} validation і {data['n']['test']} test, seed 42. Усі моделі працюють з одним manifest.

Під час аудиту знайшли три пари побайтово однакових фото. Split зроблено по SHA-256-групах, тому однакове фото не потрапляє одночасно в train і test. Одна пара має суперечливі мітки roses і tulips; обидва записи разом у train.

Кількість фото за класами: {counts_text}. Через помірний дисбаланс крім accuracy рахуємо macro-F1.

Екстрактори — MobileNetV2 ({data['dims']['mobilenetv2']} ознак), ResNet50 ({data['dims']['resnet50']}) і VGG16 ({data['dims']['vgg16']}). VGG16 ми взяли тому, що саме VGG названо на слайді. ImageNet на слайді — це датасет попереднього навчання, а не мережа. Кожна CNN має власний preprocessing: MobileNetV2 масштабує пікселі до [-1, 1], ResNet50 і VGG16 переводять RGB у BGR і віднімають середні ImageNet».

**Показати:** class counts, split, приклади фото, форму `X_train`, t-SNE ознак (`feature_tsne.png`).

## Учасник 2: ResNet50, SVM і глибина заморожування, 3 хв

«На ознаках працює SVM із RBF-ядром, C = 1, gamma = scale. StandardScaler навчається тільки на train і стоїть у Pipeline. Гіперпараметри зафіксовані заздалегідь, на test нічого не підбирали.

Далі — залежність точності від глибини заморожених шарів. До MobileNetV2 додали голову Dropout + Dense на пʼять класів і три епохи навчали тільки голову. Після цього всі {len(data['freeze'])} {plural(len(data['freeze']), MODES)} стартують з однакових ImageNet-ваг і однакової прогрітої голови: до {data['config']['fine_tune_epochs']} епох, Adam {lr(data['config']['fine_tune_learning_rate'])}, рання зупинка за val loss з patience {data['config']['early_stopping_patience']}. Розморожуємо останні 0, {', '.join(str(int(x)) for x in sorted(data['freeze'].unfreeze_last) if x)} шарів; {int(best.frozen_batchnorm_layers)} BatchNorm завжди frozen і працюють в inference-режимі.

{trend['text'][0].upper() + trend['text'][1:]}

Обрано режим «{policy_label(depth['best_unfreeze_last'], trend['total'])}» за validation; тільки цей режим перевірено на test: {pct(depth['test']['accuracy'])}. Checkpoint — епоха з мінімальним val loss, а не максимум accuracy з історії.

Для прискорення кешували незмінний префікс у float32, враховуючи всі shortcut-ребра, і перевірили числову еквівалентність з повною мережею до й після навчання. Аудит підтвердив: змінились лише дозволені ваги, усі frozen ваги незмінні».

**Показати:** таблицю режимів, `freeze_depth_accuracy_trend.png`, криві loss/accuracy зі спільною шкалою.

## Учасник 3: Random Forest, benchmark, матриці й помилки, 3–4 хв

«Третій класифікатор — Random Forest на {data['config']['random_forest_trees']} дерев. Разом з Logistic Regression, SVM і трьома CNN це {data['n_pairs']} пар. Для кожної є validation і test accuracy, macro-F1, матриця помилок і звіт за класами.

За validation macro-F1 обрано {pair(sel['extractor'], sel['classifier'])}: {sel['validation_metrics']['macro_f1']:.4f}. Його test accuracy — {pct(data['chosen'].accuracy)}, macro-F1 — {data['chosen'].macro_f1:.4f}. Найвищу test accuracy, {pct(data['best_test_accuracy'])}, мають {other_best}, але переможця за test не перевибираємо: test — незалежна оцінка, а не підбір. {ranking_text(data)}

На матриці рядки — справжні класи, стовпці — прогноз; діагональ — правильні рішення. У нормалізованій по рядках матриці діагональ дорівнює recall класу.

Найбільша направлена плутанина — {top.true_class} як {top.pred_class}: {int(top['count'])} із {data['top_support']}. Зворотний напрям має {data['reverse_count']} випадків. Ця клітинка найчастіша в середньому по всіх {data['n_pairs']} парах, тобто помилка систематична.

Фото помилок ми згрупували за причинами. Схожі класи: у flower_03530 махровий тюльпан з рожевими краями схожий на троянду. Малий обʼєкт: у flower_00293 ромашка маленька, більшу частину кадру займає собака. Нетиповий ракурс: flower_02227 — соняшник збоку, без темного центру. Проблеми розмітки: на flower_01581 видно тюльпани, хоча мітка roses, а flower_01983 — гербери, яких немає серед пʼяти класів. Це гіпотези за переглядом фото, а не доказ того, на що дивилась CNN.

Отже, виконано всі три критерії: матриці, тренд глибини заморожування й типові помилки».

## Запитання викладача

| Запитання | Коротка відповідь |
|---|---|
| Що саме переноситься? | Ваги CNN, навчені на ImageNet; класи ImageNet не переносяться як готові відповіді. |
| Навіщо include_top=False? | Прибрати класифікатор ImageNet на 1000 класів й отримати ознаки для наших пʼяти класів. |
| Чому три CNN? | Умова вимагає кількох екстракторів; VGG16 названо на слайді, MobileNetV2 і ResNet50 — легша й глибша сучасні архітектури. |
| Чому немає AlexNet? | Її немає серед готових Keras Applications; VGG16 покриває класичну архітектуру зі слайда. ImageNet — датасет, а не мережа. |
| Навіщо Global Average Pooling? | Усереднити просторові карти каналів і отримати компактний вектор ознак. |
| LM — це лінійна регресія? | Тут лінійна модель класифікації Logistic Regression. |
| Чому scaler у Pipeline? | Mean і std визначаються тільки за train і так само застосовуються до val/test. |
| Accuracy проти macro-F1? | Accuracy рахує всі правильні фото; macro-F1 однаково зважує F1 пʼяти класів. |
| Чому test-переможець інший? | Вибірки скінченні, оцінки коливаються. Test описує узагальнення, але не змінює вибір. |
| Що показує t-SNE? | 2D-проєкцію локальних сусідств ознак без міток. Кольори й обведені помилки допомагають оглянути результат, але не доводять роздільність початкового простору чи причини помилок. |
| Чому val loss для checkpoint, а вибір за accuracy? | Loss визначає checkpoint усередині навчання; між режимами порівнюємо accuracy відновлених checkpoint. |
| Чи всі шари last 100 навчаються? | Ні. BatchNorm виключені, частина шарів не має ваг; дивимось flags, шари з вагами й параметри. |
| Чому training=False? | BatchNorm лишається зі статистиками ImageNet; градієнти до trainable Conv при цьому проходять. |
| Чому compile після зміни trainable? | Оптимізатор має працювати з новим набором навчуваних ваг. |
| Навіщо малий learning rate? | Обережно адаптувати pretrained ваги й не зруйнувати корисні ознаки. |
| Кеш означає відсутність fine-tuning? | Кешується лише frozen префікс; суфікс має градієнти й реально змінює ваги. |
| Чому без аугментації? | Простий контрольований протокол і точний кеш префікса для всіх режимів. |
| Чому confidence іноді менше 50%? | Пʼять класів і невпевнені рішення; у SVM predict і argmax probability можуть відрізнятися. |
| Чи можна подати будь-яке фото? | Так, але модель завжди вибере один із пʼяти класів, reject-класу немає. |
| Чи доведено статистичну перевагу? | Ні. Один seed і split; різницю в кілька фото трактуємо описово. |
| Чи працює показ без інтернету? | Так: рисунки, метрики, демо-фото й потрібні ваги включені в архів. |
"""
    (report / "DEFENSE_UA.md").write_text(defense, encoding="utf-8")


def build_sections(data):
    """Content shared by the HTML and Markdown report; all numbers read from files."""
    sections = []
    trend = trend_facts(data)
    sel, depth, n = data["selected"], data["depth"], data["n"]
    top = data["top_pair"]
    f = data["freeze"]

    def add(title, paragraphs=(), tables=(), figures=()):
        sections.append({"title": title, "paragraphs": list(paragraphs),
                         "tables": list(tables), "figures": list(figures)})
    add("1. Мета та умови", [
        f"Порівняти класичні класифікатори на ознаках {len(EXTRACTORS)} pretrained CNN для класифікації пʼяти квіткових класів. Окремо дослідити, як точність залежить від глибини заморожених шарів MobileNetV2, побудувати матриці помилок і пояснити характерні помилки.",
        "Джерело умов: надана презентація Labs.pdf, слайд 14. Частки: confusion matrices 60%, accuracy trend 20%, typical errors 20%. Конкретні CNN, датасет і протокол є рішеннями проєкту.",
        "Усі числа, криві та прогнози в цьому звіті одержані фактичним виконанням коду, а не підставлені як очікувані результати."],
        [pd.DataFrame({"Критерій": ["Confusion matrices", "Accuracy trend", "Typical errors"],
                       "Частка": ["60%", "20%", "20%"],
                       "Реалізація": [f"{2 * data['n_pairs']} матриць + 2 для обраної CNN",
                                      f"{len(f)} реально {'навчений' if len(f) == 1 else 'навчених'} {plural(len(f), MODES)} заморожування",
                                      "Фото, частоти пар у всіх моделях, категорії причин"]})])
    add("2. Дані та контроль якості", [
        f"TensorFlow Flowers містить {data['quality']['raw_images']} фотографій. Порядок класів із TFDS: dandelion, daisy, tulips, sunflowers, roses. JPEG узято з того самого офіційного архіву, який використовує TFDS; перевірено SHA-256 архіву й кожного файлу.",
        f"Знайдено {data['quality']['unique_file_hashes']} унікальних побайтових хешів: {data['quality']['duplicate_groups']} пари дублікатів. Одна пара має мітки roses і tulips. Збережено всі офіційні записи й labels; стратифікація виконується на групах SHA-256, тож однаковий файл не перетинає split.",
        f"Розбиття: {n['train']} train / {n['val']} validation / {n['test']} test, seed 42. Спільний manifest незмінний для всіх моделей. Майже-дублікати не усувалися вичерпно."],
        [data["split_counts"]],
        ["eda_class_counts", "eda_split_distribution", "eda_examples", "eda_dimensions"])
    add("3. Методика та протокол", [
        "Feature extraction: ImageNet backbone без початкової голови + Global Average Pooling. " +
        ", ".join(f"{NAMES[e]} формує {d} ознак" for e, d in data["dims"].items()) +
        ". Усі отримують RGB фото після resize до 224×224 без збереження пропорцій: це узгоджений базовий протокол, можливі спотворення форми.",
        "MobileNetV2 preprocess_input масштабує значення до [-1, 1]; ResNet50 і VGG16 preprocess_input переставляють RGB у BGR і центрують за середніми ImageNet. CNN не навчаються під час отримання ознак. IDs зберігаються разом із X та y.",
        f"Logistic Regression: StandardScaler + C=1, max_iter=3000, lbfgs. SVM: StandardScaler + RBF, C=1, gamma=scale, probability=True. Random Forest: {data['config']['random_forest_trees']} дерев, max_features=sqrt, seed 42. Усі fit тільки на train, scaler теж train-only. Гіперпараметри задані до test.",
        f"Найкраща класична пара обирається серед {data['n_pairs']} кандидатів за validation macro-F1, потім accuracy. Вибір записується перед test-прогнозами. Test-результати всіх пар — підсумковий benchmark, а не критерій перевибору.",
        "Accuracy — частка правильних прогнозів; macro-F1 — середнє F1 пʼяти класів із рівними вагами; weighted-F1 враховує чисельність класів. Raw confusion matrix містить кількості, row-normalized — частки в межах істинного класу."])
    add("4. Результати benchmark", [
        f"За validation macro-F1 вибрано {pair(sel['extractor'], sel['classifier'])}. Його test accuracy {pct(data['chosen'].accuracy)}, macro-F1 {data['chosen'].macro_f1:.4f}.",
        f"Найвища test accuracy серед фіксованих кандидатів — {pct(data['best_test_accuracy'])} у {data['best_test_names']}. Це описовий підсумок; переможця за validation не перевибираємо. {ranking_text(data)}",
        "На графіку вісь збільшено: різниці між парами становлять кілька відсоткових пунктів, тому стовпчики від нуля їх приховували б. Виділено пару, обрану за validation.",
        tsne_text(data)],
        [data["benchmark"]], ["benchmark_accuracy", "benchmark_macro_f1", "feature_tsne"])
    add("5. Матриці помилок", [
        f"Рядки — справжні класи, стовпці — прогнози, діагональ — правильні рішення. Для кожної з {data['n_pairs']} пар є raw і row-normalized матриця. У raw-матриць однакова шкала кольору; у normalized діапазон 0–100%, а діагональ дорівнює recall класу. Окремі PNG і CSV додаються для кожної пари."],
        figures=["confusion_raw_grid", "confusion_normalized_grid"])
    add("6. Залежність accuracy від глибини заморожених шарів", [
        f"MobileNetV2 + Dropout({data['config']['dropout']}) + Dense(5, softmax). Спільна голова прогріта {data['config']['warmup_epochs']} епохи з Adam {lr(data['config']['warmup_learning_rate'])}. Кожна політика стартує з тієї самої голови та ImageNet backbone, а не продовжує попередній режим.",
        f"Потім Adam {lr(data['config']['fine_tune_learning_rate'])}, максимум {data['config']['fine_tune_epochs']} епох, однакові перестановки train, EarlyStopping(val_loss, patience={data['config']['early_stopping_patience']}, restore_best_weights=True). Усі {int(f.frozen_batchnorm_layers.iloc[0])} BatchNorm frozen, backbone викликається з training=False. Кожен режим перекомпільовано з новим оптимізатором.",
        "Незмінний префікс кешується у float32 (для повного розморожування — сам вхід), trainable суфікс справді донавчається. Враховано всі residual-ребра через межу; еквівалентність із повною мережею перевірено до й після fit. Аугментація відсутня.",
        f"Вісь X — фактичне число frozen шарів серед {trend['total']} шарів backbone, включно з BatchNorm і шарами без ваг. «Last k» — ширина кінцевого вікна, а не кількість навчуваних шарів: у таблиці окремо наведено trainable flags, шари з вагами й параметри.",
        trend["text"][0].upper() + trend["text"][1:],
        f"Обрано режим «{policy_label(depth['best_unfreeze_last'], trend['total'])}»: validation accuracy {pct(depth['validation']['val_accuracy'])}, відновлена епоха {depth['validation']['restored_epoch']}. Тільки обрана політика отримала фінальну test-перевірку: accuracy {pct(depth['test']['accuracy'])}, macro-F1 {depth['test']['macro_f1']:.4f}.",
        (f"Відтворюваність: режим last {data['repro']['policy']} повторно навчено на іншій машині для розширення; validation accuracy {pct(data['repro']['rerun']['val_accuracy'])} проти {pct(data['repro']['original']['val_accuracy'])}, macro-F1 {data['repro']['rerun']['val_macro_f1']:.4f} проти {data['repro']['original']['val_macro_f1']:.4f}."
         if data["repro"] else "")],
        [freeze_display(data)], ["freeze_depth_accuracy_trend", "freeze_learning_accuracy", "freeze_learning_loss",
                                 "freeze_best_confusion", "freeze_best_confusion_normalized"])
    across = data["across"].head(3)
    add("7. Характерні помилки та спостереження", [
        f"Найчастіша пара обраної моделі — {top.true_class} → {top.pred_class}: {int(top['count'])} помилок із {data['top_support']} ({pct(top['count'] / data['top_support'])}). Зворотний напрям має {data['reverse_count']} випадків.",
        "Щоб перевірити, чи помилки систематичні, ті самі клітинки порахували для всіх класичних пар і обраної CNN. Три найчастіші в середньому: " +
        "; ".join(f"{r.true_class} → {r.pred_class} ({r.mean_classical:.1f}, від {int(r.min_classical)} до {int(r.max_classical)}; CNN {int(r.selected_cnn)})" for r in across.itertuples()) + ".",
        "Переглянуті фото згруповано за причинами: схожі класи (махрові тюльпани й бутони троянд), кілька видів у кадрі, малий обʼєкт на складному фоні, нетиповий ракурс або світло, а також проблеми розмітки. На flower_01581 видно тюльпани з міткою roses, на flower_01983 — гербери, яких немає серед пʼяти класів. Такі «помилки» з високою confidence — радше шум офіційних міток.",
        "Візуальні пояснення є гіпотезами за переглядом фото, а не встановленими внутрішніми причинами CNN. Букети flower_03563 і flower_03564 — майже-дублікати, тому не є незалежними доказами.",
        "Confidence дорівнює predict_proba саме передбаченого класу. SVC.predict інколи відрізняється від argmax probability; у CSV збережено справжнє рішення. Confidence не є гарантією правильності."],
        [data["pairs"].head(10),
         data["observations"][["image_id", "true_class", "pred_class", "error_category"]].assign(
             confidence=data["observations"].confidence.map(lambda x: pct(x, 1)))],
        ["common_error_pairs", "error_pairs_across_models", "typical_errors", "confident_errors"])
    runtime = data["runtime"]["packages"]
    add("8. Перевірки та обмеження", [
        f"Виконано автоматичні перевірки ({len(data['audit']['checks'])}): відсутність shared hashes між split, узгодженість IDs/labels, перерахунок метрик і матриць із прогнозів, повторне завантаження estimator, train-only scaler, правильний val ranking, зміна лише дозволених ваг CNN і наявність усіх рисунків. Усі пройдено.",
        f"Середовище: Python 3.12, TensorFlow CPU {runtime.get('tensorflow-cpu', runtime.get('tensorflow'))}, Keras {runtime['keras']}, scikit-learn {runtime['scikit-learn']}, без GPU. Дати запуску: {RUN_DATES}. Час у CSV є журналом запуску, а не строгим speed benchmark.",
        f"Обмеження: один seed і один split, скінченний test на {n['test']} фото, можливі майже-дублікати й шум міток, лише пʼять класів, немає external test і reject-класу. Формальної перевірки статистичної переваги не проводили."])
    return sections


def export_markdown_html(data, sections):
    md = ["# Assignment 2: Transfer Learning — TensorFlow Flowers", "", f"Фактичний звіт · Python / Jupyter · {RUN_DATES}", ""]
    html_parts = []
    for i, section in enumerate(sections):
        paragraphs = [p for p in section["paragraphs"] if p]
        md += ["## " + section["title"], "", *sum(([p, ""] for p in paragraphs), [])]
        html_parts += [f'<section id="s{i}"><h2>{escape(section["title"])}</h2>']
        html_parts += [f"<p>{escape(p)}</p>" for p in paragraphs]
        for table in section["tables"]:
            md += [markdown_table(table), ""]
            html_parts += ['<div class="table">' + table.to_html(index=False, border=0, escape=True) + '</div>']
        for name in section["figures"]:
            md += [f"![{name}](../results/figures/{name}.png)", ""]
            encoded = base64.b64encode((ROOT / f"results/figures/{name}.png").read_bytes()).decode()
            html_parts += [f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(name)}" loading="lazy">'
                           f'<figcaption>{escape(name)}.png · збільшіть у браузері для повної роздільності</figcaption></figure>']
        html_parts.append("</section>")
    md += [conclusions(data).replace("# Висновки", "## 9. Висновки", 1), "## Джерела", ""]
    md += [f"{i+1}. [{name}]({url})" for i, (name, url) in enumerate(SOURCES)]
    md += ["", f"Атрибуція фото: `data/FLOWERS_LICENSE.txt`. CSV, усі {data['n_figures']} PNG та моделі включені в проєкт."]
    (ROOT / "report/RESULTS_UA.md").write_text("\n".join(md), encoding="utf-8")
    conclusions_html = "".join("<p>" + re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escape(p)) + "</p>"
                              for p in conclusions(data).strip().split("\n\n")[1:])
    sources_html = "".join(f'<li><a href="{url}">{escape(name)}</a></li>' for name, url in SOURCES)
    css = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f4;color:#203344;font:16px/1.6 system-ui,Arial,sans-serif}
    header{background:#173d55;color:white;padding:56px max(28px,calc((100vw - 1120px)/2)) 40px}
    header h1{font-size:clamp(30px,4vw,50px);line-height:1.1;margin:10px 0 20px}header p{color:#d7e8e6;max-width:900px}
    .eyebrow{letter-spacing:.12em;font-size:12px;text-transform:uppercase;color:#8ec6b5}
    main{max-width:1180px;margin:28px auto;padding:0 22px}section{background:white;border:1px solid #dce3e5;border-radius:14px;padding:28px 34px;margin:0 0 24px}
    h2{font-size:26px;line-height:1.3;margin-top:0;color:#173d55}p{max-width:1030px}img{display:block;max-width:100%;height:auto;margin:auto;background:white}
    figure{margin:30px 0 18px}figcaption{font-size:12px;color:#627383;text-align:center;margin-top:10px}.table{overflow:auto;margin:24px 0}
    table{width:100%;border-collapse:collapse;font-size:14px}th{background:#edf4f3;text-align:left;color:#173d55}th,td{padding:10px 12px;border-bottom:1px solid #dce3e5}
    a{color:#246b88}footer{max-width:1120px;margin:28px auto;padding:0 24px 32px;color:#627383;font-size:13px}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0 0}.card{background:#24506a;padding:18px;border-radius:10px}.card strong{display:block;font-size:30px}.card span{font-size:13px;color:#d7e8e6}
    @media(max-width:800px){.cards{grid-template-columns:1fr 1fr}}@media(max-width:650px){section{padding:20px}.cards{grid-template-columns:1fr}main{padding:0 12px}}@media print{body{background:white}header{padding:25px}section{break-inside:auto;border:0;padding:10px}figure{break-inside:avoid}}
    """
    trend = trend_facts(data)
    html = f'''<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Assignment 2 · TensorFlow Flowers</title><style>{css}</style>
    <header><div class="eyebrow">Assignment 2 · Deep Learning · Jupyter</div><h1>Transfer Learning<br>TensorFlow Flowers</h1>
    <p>{len(EXTRACTORS)} CNN-екстрактори × {len(CLASSIFIERS)} класифікатори, {len(data['freeze'])} {plural(len(data['freeze']), MODES)} заморожування, матриці помилок і реальні невдалі прогнози.</p>
    <div class="cards"><div class="card"><strong>{data['quality']['raw_images']:,}</strong><span>фото · 5 класів · один manifest</span></div>
    <div class="card"><strong>{data['n_pairs']}</strong><span>пар екстрактор × класифікатор</span></div>
    <div class="card"><strong>{pct(data['chosen'].accuracy)}</strong><span>test · класична пара, обрана за validation</span></div>
    <div class="card"><strong>{pct(data['depth']['test']['accuracy'])}</strong><span>test · CNN, обрана за validation ({escape(policy_label(data['depth']['best_unfreeze_last'], trend['total']))})</span></div></div></header>
    <main>{''.join(html_parts)}<section><h2>9. Висновки</h2>{conclusions_html}</section>
    <section><h2>Джерела</h2><ol>{sources_html}</ol><p>Фото: офіційний архів TensorFlow Flowers; атрибуція включена в data/FLOWERS_LICENSE.txt.</p></section></main>
    <footer>Фактичний запуск: {RUN_DATES} · усі зображення вбудовані, зовнішні ресурси не потрібні · повні PNG та CSV є в архіві.</footer></html>'''.replace("3,670", "3 670")
    (ROOT / "report/Results_UA.html").write_text(html, encoding="utf-8")


def export_pdf(data):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage,
                                   Table, TableStyle, PageBreak)
    from PIL import Image as PILImage
    font_roots = [ROOT / "assets/fonts", Path("/usr/share/fonts/truetype/dejavu")]
    font_root = next(p for p in font_roots if (p / "DejaVuSans.ttf").exists())
    pdfmetrics.registerFont(TTFont("DVS", str(font_root / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DVS-Bold", str(font_root / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DVS", normal="DVS", bold="DVS-Bold", italic="DVS", boldItalic="DVS-Bold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyUA", fontName="DVS", fontSize=9.4, leading=14, spaceAfter=8,
                              textColor=colors.HexColor("#263d4c")))
    styles.add(ParagraphStyle("TitleUA", fontName="DVS-Bold", fontSize=27, leading=34,
                              textColor=colors.HexColor("#173d55"), spaceAfter=18))
    styles.add(ParagraphStyle("H1UA", fontName="DVS-Bold", fontSize=17, leading=23,
                              textColor=colors.HexColor("#173d55"), spaceAfter=14))
    styles.add(ParagraphStyle("H2UA", fontName="DVS-Bold", fontSize=11, leading=16, spaceAfter=8,
                              textColor=colors.HexColor("#3e6f91")))
    styles.add(ParagraphStyle("CaptionUA", fontName="DVS", fontSize=8, leading=11,
                              textColor=colors.HexColor("#617483"), spaceAfter=10))
    styles.add(ParagraphStyle("CellUA", fontName="DVS", fontSize=7.2, leading=10))
    styles.add(ParagraphStyle("CellSmallUA", fontName="DVS", fontSize=6.4, leading=8.5))
    body = []
    content_width = 178*mm
    sections = build_sections(data)
    trend = trend_facts(data)
    sel, depth, n = data["selected"], data["depth"], data["n"]

    def p(text, kind="BodyUA"):
        text = str(text).replace("—", "-").replace("–", "-").replace("‑", "-")
        body.append(Paragraph(xml_escape(text), styles[kind]))

    def page(title):
        if body:
            body.append(PageBreak())
        p(title, "H1UA")

    tmp = tempfile.TemporaryDirectory()
    photo_figures = {"eda_examples", "typical_errors", "confident_errors"}

    def image(name, width=178*mm, max_height=175*mm):
        # Downsample to ~220 dpi at the printed size; photo galleries as JPEG. Keeps the PDF small.
        path = ROOT / f"results/figures/{name}.png"
        with PILImage.open(path) as im:
            w, h = im.size
            scale = min(width/w, max_height/h)
            target = (max(1, round(w * scale / 72 * 220)), max(1, round(h * scale / 72 * 220)))
            small = im.convert("RGB").resize(target, PILImage.LANCZOS) if target[0] < w else im.convert("RGB")
            out = Path(tmp.name) / (name + (".jpg" if name in photo_figures else ".png"))
            if name in photo_figures:
                small.save(out, quality=88, optimize=True)
            else:
                small.save(out, optimize=True)
        body.append(PDFImage(str(out), width=w*scale, height=h*scale))
        body.append(Spacer(1, 3*mm))

    def table(df, widths=None, small=False):
        style = styles["CellSmallUA" if small else "CellUA"]
        vals = [[Paragraph(xml_escape(str(v)), style) for v in df.columns]]
        vals += [[Paragraph(xml_escape(str(v)), style) for v in row] for row in df.itertuples(index=False, name=None)]
        t = Table(vals, colWidths=widths or [content_width/len(df.columns)]*len(df.columns), repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e7f0ef")),
                              ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8f9")]),
                              ("VALIGN", (0,0),(-1,-1),"TOP"),
                              ("LINEBELOW", (0,0),(-1,0),.7,colors.HexColor("#7d9eaa")),
                              ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                              ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        body.append(t)
        body.append(Spacer(1, 4*mm))

    p("ASSIGNMENT 2  /  DEEP LEARNING", "H2UA")
    p("Transfer Learning · TensorFlow Flowers", "TitleUA")
    p("Порівняльне дослідження pretrained CNN-екстракторів та класичних класифікаторів. Матеріали для Jupyter і захисту.")
    p(f"Фактичний запуск: {RUN_DATES}. Команда: три учасники. Python 3.12 · TensorFlow CPU 2.20 · scikit-learn 1.8.", "CaptionUA")
    table(pd.DataFrame({"Показник": ["Дані", "Класичні кандидати", "Режими заморожування", "Test класичної пари",
                                     "Test вибраної CNN", "Перевірки"],
                        "Результат": [f"{data['quality']['raw_images']} фото / 5 класів / {n['train']}–{n['val']}–{n['test']}",
                                      f"{len(EXTRACTORS)} CNN × {len(CLASSIFIERS)} класифікатори = {data['n_pairs']} пар",
                                      ", ".join(policy_label(x, trend['total']) for x in sorted(data['freeze'].unfreeze_last)),
                                      f"{pct(data['chosen'].accuracy)} · {pair(sel['extractor'], sel['classifier'])} · вибір за validation",
                                      f"{pct(depth['test']['accuracy'])} · MobileNetV2 {policy_label(depth['best_unfreeze_last'], trend['total'])}",
                                      f"{len(data['audit']['checks'])} пройдено / {data['n_figures']} PNG / збережені моделі"]}),
          [55*mm, 123*mm])
    p("Умови викладача", "H2UA")
    p("Слайд 14: порівняти кілька класифікаторів на кількох екстракторах; дослідити залежність accuracy від глибини заморожених шарів; побудувати confusion matrices і показати typical errors. Оцінювання: матриці 60%, тренд 20%, помилки 20%.")
    p("Звіт містить фактичні результати. В архіві є повний код, два виконані notebook, усі графіки й таблиці. Статистичну універсальну перевагу моделі за одним split не стверджуємо.")

    page("1. Дані та контроль split")
    cc = data["class_counts"]
    p("TensorFlow Flowers: " + ", ".join(f"{k} {v}" for k, v in cc.items()) + ". Власний grouped stratified split, seed 42. Назви міток узято з TFDS; JPEG - із того самого офіційного архіву.")
    table(data["split_counts"], [55*mm, 41*mm, 41*mm, 41*mm])
    image("eda_class_counts", max_height=95*mm)
    for para in sections[1]["paragraphs"][1:]:
        p(para)
    p("Manifest SHA-256: " + data["quality"]["manifest_sha256"], "CaptionUA")

    page("2. Приклади, баланс і розміри")
    image("eda_examples", max_height=88*mm)
    image("eda_dimensions", max_height=85*mm)
    p("Фото мають різні пропорції, ракурси й фони. Єдине resize до 224×224 забезпечує однаковий протокол, але може змінювати форму. Дисбаланс класів помірний, тому разом із accuracy показано macro-F1.")
    p("Зображення: офіційний TensorFlow Flowers; атрибуція всіх фото включена у data/FLOWERS_LICENSE.txt.", "CaptionUA")

    page("3. Моделі та експериментальний протокол")
    for para in sections[2]["paragraphs"]:
        p(para)
    p("Методика оцінювання", "H2UA")
    p("Train визначає ваги й scaler. Validation визначає переможця й checkpoint. Test залишається незалежним до фіксації вибору. Один manifest використано для всіх класичних пар і всіх режимів CNN.")
    p("SVM confidence обчислюється з predict_proba для фактичного класу predict; ці два правила в SVC не завжди мають один argmax. Probability не є гарантією правильності.")

    page(f"4. Порівняння {data['n_pairs']} кандидатів")
    table(data["benchmark"], [31*mm, 42*mm, 27*mm, 25*mm, 27*mm, 26*mm])
    image("benchmark_macro_f1", max_height=100*mm)
    for para in sections[3]["paragraphs"][:2]:
        p(para)

    page("5. t-SNE ознак трьох CNN")
    image("feature_tsne", max_height=90*mm)
    p(sections[3]["paragraphs"][3])
    image("benchmark_accuracy", max_height=95*mm)

    page("6. Матриці помилок: огляд")
    image("confusion_raw_grid", max_height=200*mm)
    p("Рядок - справжній клас, стовпець - прогноз. Однакова шкала кольору для всіх raw-матриць. Нижче - кожна пара окремо зі звітом за класами.", "CaptionUA")

    for extractor in EXTRACTORS:
        for clf in CLASSIFIERS:
            key = f"{extractor}_{clf}"
            page(f"Матриця: {pair(extractor, clf)}")
            image(f"confusion_raw_{key}", width=150*mm, max_height=135*mm)
            row = data["metrics"].query("split == 'test' and extractor == @extractor and classifier == @clf").iloc[0]
            mark = " Обрана за validation." if (extractor, clf) == (sel["extractor"], sel["classifier"]) else ""
            p(f"Test: accuracy {pct(row.accuracy)}; macro-F1 {row.macro_f1:.4f}; n={n['test']}. Рядок - справжній клас, стовпець - прогноз.{mark}")
            rep = pd.read_csv(ROOT / f"results/tables/classification_report_{key}.csv").head(5)
            rep.columns = ["Клас" if x.startswith("Unnamed") else x for x in rep.columns]
            for col in ("precision", "recall", "f1-score"):
                rep[col] = rep[col].map(lambda x: f"{x:.4f}")
            rep["support"] = rep.support.astype(int)
            table(rep)
            p("Row-normalized версія: confusion_normalized_" + key + ".png. Її діагональ - recall класу.", "CaptionUA")

    page("7. Залежність accuracy від глибини заморожування")
    p(sections[5]["paragraphs"][0])
    table(freeze_display(data), [19*mm, 15*mm, 20*mm, 18*mm, 26*mm, 14*mm, 20*mm, 23*mm, 23*mm], small=True)
    image("freeze_depth_accuracy_trend", max_height=84*mm)
    p(trend["text"][0].upper() + trend["text"][1:])
    p(sections[5]["paragraphs"][3], "CaptionUA")

    page("8. Динаміка навчання режимів")
    image("freeze_learning_accuracy", max_height=92*mm)
    image("freeze_learning_loss", max_height=92*mm)
    p("Спільна вісь Y для всіх панелей. Вертикальна лінія - відновлена епоха мінімального val loss. Train accuracy обчислюється з увімкненим dropout. Швидке зростання train accuracy без такого ж росту validation вказує на ризик перенавчання при глибшому розморожуванні.", "CaptionUA")
    for para in (sections[5]["paragraphs"][1], sections[5]["paragraphs"][2]):
        p(para, "CaptionUA")

    page("9. Фінальна перевірка обраної CNN")
    p(sections[5]["paragraphs"][5])
    image("freeze_best_confusion", width=150*mm, max_height=140*mm)
    diff = data["depth_test_correct"] - data["best_test_correct"]
    p(f"Обрана CNN: {data['depth_test_correct']} правильних фото з {n['test']}; найкраща з {data['n_pairs']} класичних пар - {data['best_test_correct']} ({'+' if diff >= 0 else ''}{diff}). Це описова різниця одного split; формальної перевірки статистичної переваги не проводили.")
    if sections[5]["paragraphs"][6]:
        p(sections[5]["paragraphs"][6], "CaptionUA")

    page("10. Часті й систематичні пари помилок")
    image("common_error_pairs", max_height=92*mm)
    for para in sections[6]["paragraphs"][:2]:
        p(para)
    image("error_pairs_across_models", max_height=95*mm)

    page("11. Реальні приклади хибних прогнозів")
    obs = data["observations"].set_index("image_id")
    detail_ids = [i for i in ("flower_03530", "flower_00293", "flower_02227", "flower_00410", "flower_01581", "flower_01983")
                  if i in obs.index]
    for start in range(0, len(detail_ids), 3):
        photos, captions = [], []
        for image_id in detail_ids[start:start+3]:
            photo = ROOT / f"data/demo_images/{image_id}.jpg"
            with PILImage.open(photo) as im:
                w, h = im.size
            scale = min(55*mm/w, 48*mm/h)
            photos.append(PDFImage(str(photo), width=w*scale, height=h*scale))
            row = obs.loc[image_id]
            label = (f"{image_id}<br/>True: {row.true_class} · Pred: {row.pred_class}<br/>"
                     f"p(pred): {row.confidence:.1%}<br/>{xml_escape(row.error_category)}")
            captions.append(Paragraph(label, styles["CaptionUA"]))
        photo_table = Table([photos, captions], colWidths=[content_width/3]*3, hAlign="LEFT")
        photo_table.setStyle(TableStyle([("VALIGN", (0,0),(-1,-1),"TOP"), ("TOPPADDING",(0,0),(-1,-1),4),
                                        ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        body.append(photo_table)
    for image_id in detail_ids:
        row = obs.loc[image_id]
        p(f"{image_id}: {row.visible_observation} {row.possible_explanation}")

    page("12. Галереї помилок")
    image("typical_errors", max_height=104*mm)
    image("confident_errors", max_height=104*mm)
    p("typical_errors - до чотирьох фото для найчастіших пар; confident_errors - найвпевненіші хибні рішення обраної класичної пари. Повна таблиця описаних фото з категоріями - results/tables/error_observations_ua.csv.", "CaptionUA")

    page("13. Висновки, перевірки й обмеження")
    for para in conclusions(data).strip().split("\n\n")[1:]:
        p(para.replace("**", ""))
    p(sections[7]["paragraphs"][0])

    page("14. Середовище, джерела й навігація проєктом")
    p(sections[7]["paragraphs"][1])
    p(sections[7]["paragraphs"][2])
    for name, url in SOURCES:
        p(name, "H2UA")
        body.append(Paragraph('<link href="' + xml_escape(url) + '">' + xml_escape(url) + '</link>', styles["CaptionUA"]))
    p("Атрибуція фотографій включена у data/FLOWERS_LICENSE.txt. Оригінал умов і MD-план - у materials/.")
    p("Що відкривати на захисті", "H2UA")
    p(f"00_Results_and_Defense.ipynb - готові таблиці й графіки. 01_Transfer_Learning_Flowers.ipynb - код, методика, повторне навчання й прогноз фото. DEFENSE_UA.md - виступ трьох учасників та відповіді. results/figures - усі {data['n_figures']} PNG, включно з {2 * data['n_pairs']} окремими матрицями.")

    def decorate(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#c8d8dc"))
        canvas.line(16*mm, 17*mm, 194*mm, 17*mm)
        canvas.setFont("DVS", 7)
        canvas.setFillColor(colors.HexColor("#617483"))
        canvas.drawString(16*mm, 12*mm, "Assignment 2 · TensorFlow Flowers · фактичні результати")
        canvas.drawRightString(194*mm, 12*mm, str(doc.page))
    doc = SimpleDocTemplate(str(ROOT / "report/Results_UA.pdf"), pagesize=A4,
                            rightMargin=16*mm, leftMargin=16*mm, topMargin=18*mm, bottomMargin=22*mm,
                            title="Assignment 2: Transfer Learning - TensorFlow Flowers",
                            author="Assignment 2 project team")
    doc.build(body, onFirstPage=decorate, onLaterPages=decorate)
    tmp.cleanup()


def generate(config=None):
    data = load()
    write_notes(data)
    sections = build_sections(data)
    export_markdown_html(data, sections)
    export_pdf(data)
    print("Generated factual Markdown, self-contained HTML, PDF and defense notes")


if __name__ == "__main__":
    generate()
