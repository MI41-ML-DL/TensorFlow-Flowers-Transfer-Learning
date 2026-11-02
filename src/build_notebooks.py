"""Build portable Ukrainian Jupyter notebooks; execution is a separate step."""
from pathlib import Path
import nbformat as nbf

from .common import CLASSIFIERS, EXTRACTORS, ROOT, read_config
from .visuals import expected_figures


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


INITIAL = '''
from pathlib import Path
import json
import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image

# Працює з кореня проєкту або папки notebooks; абсолютних шляхів автора немає.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "configs/full.json").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Розпакуйте весь проєкт і відкрийте ноутбук із його папки.")
NAMES = {"mobilenetv2": "MobileNetV2", "resnet50": "ResNet50", "vgg16": "VGG16",
         "logreg": "Logistic Regression", "svm": "SVM RBF", "randomforest": "Random Forest"}
def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))
def table(name, **kwargs):
    return pd.read_csv(ROOT / "results/tables" / name, **kwargs)
def figure(name, width=1050):
    display(Image(filename=str(ROOT / "results/figures" / (name + ".png")), width=width))

config = read_json("configs/full.json")
quality = read_json("results/data_quality.json")
selected = read_json("results/selected_benchmark.json")
depth = read_json("results/selected_depth_test.json")
metrics = table("benchmark_metrics.csv")
print("Проєкт знайдено. Готові результати завантажено; навчання не запускається.")
'''


def summary_code(n_figures):
    return '''
chosen = metrics.query("split == 'test'")
chosen = chosen[(chosen.extractor == selected["extractor"]) &
                (chosen.classifier == selected["classifier"])].iloc[0]
n_pairs = len(metrics.query("split == 'test'"))
policy = "усі шари" if depth["best_unfreeze_last"] >= depth["validation"]["total_backbone_layers"] \\
    else f"last {depth['best_unfreeze_last']}"
display(Markdown(f"""
**Виконано:** {quality['raw_images']} фото, 5 класів, {n_pairs} класичних комбінацій; режимів заморожування CNN — {len(config['unfreeze_last'])}.

* Класична пара, обрана за validation: **{NAMES[selected['extractor']]} + {NAMES[selected['classifier']]}**.
  Test accuracy **{chosen.accuracy:.2%}**, macro-F1 **{chosen.macro_f1:.4f}**.
* CNN, обрана за validation: **MobileNetV2, {policy}**.
  Test accuracy **{depth['test']['accuracy']:.2%}**, macro-F1 **{depth['test']['macro_f1']:.4f}**.
* Дані: **{quality['split_counts']['train']} train / {quality['split_counts']['val']} validation /
  {quality['split_counts']['test']} test**. Seed 42.
* В архіві **''' + str(n_figures) + ''' PNG**, CSV із рішеннями для кожного тестового фото, збережені моделі,
  історії навчання, звіт і сценарій захисту.

Числа нижче є результатами фактичного запуску. Test не використовувався для вибору моделі.
"""))
'''


def shared_results(config):
    n_pairs = len(EXTRACTORS) * len(CLASSIFIERS)
    policies = ", ".join(str(x) for x in config["unfreeze_last"])
    return [
        md("""## 1. Дані й відтворюваність

Офіційний набір має єдину частину `train`; власні train/validation/test створено один раз.
Порядок класів узято з TFDS, він не алфавітний. Фото однакового SHA-256 залишені в одному split.
Три пари точних дублікатів збережені, одна пара має суперечливі мітки `roses` / `tulips`.
У цьому розбитті всі три пари потрапили до train. Майже однакові кадри окремо не усувалися.
"""),
        code('display(pd.DataFrame({"label": range(5), "class_name": read_json("data/class_names.json")}))\n'
             'display(table("split_counts.csv"))\n'
             'display(table("duplicate_audit.csv")[["image_id", "class_name", "split", "conflicting_labels"]])\n'
             'print("Manifest SHA-256:", quality["manifest_sha256"])'),
        code('figure("eda_class_counts")\nfigure("eda_split_distribution")'),
        code('figure("eda_examples", 1200)\nfigure("eda_dimensions")'),
        md(f"""## 2. Benchmark: {len(EXTRACTORS)} екстрактори × {len(CLASSIFIERS)} класифікатори = {n_pairs} пар

ImageNet CNN працює як незмінний екстрактор: `include_top=False`, Global Average Pooling.
MobileNetV2 дає 1280 ознак, ResNet50 — 2048, VGG16 — 512. VGG16 додано, бо саме VGG названо на слайді.
Logistic Regression і SVM навчаються в Pipeline зі StandardScaler; scaler бачить тільки train.
Random Forest не масштабується. Моделі й гіперпараметри визначені заздалегідь, Grid Search на test не проводився.

**Правило вибору:** найбільший validation macro-F1, потім validation accuracy.
Після фіксації переможця показано test для всіх {n_pairs} заздалегідь заданих кандидатів.
Найвища test-цифра серед кандидатів не змінює зафіксований вибір.
"""),
        code('comparison = metrics.pivot(index=["extractor", "classifier"], columns="split", '
             'values=["accuracy", "macro_f1"])\ndisplay(comparison.round(4))\n'
             'print("Вибір за validation:", NAMES[selected["extractor"]], "+", NAMES[selected["classifier"]])'),
        code('figure("benchmark_accuracy")\nfigure("benchmark_macro_f1")'),
        md("""### t-SNE ознак (візуалізація, як на слайді 14)

Тестові ознаки кожної CNN проєктуються у 2D без використання міток; колір — справжній клас.
Компактні кластери ілюструють локальні сусідства у 2D; це не доказ роздільності в початковому просторі ознак.
Кружками позначено тестові помилки обраної класичної пари. Їхнє положення в t-SNE не доводить причину помилки.
"""),
        code('figure("feature_tsne", 1400)'),
        md(f"""## 3. Матриці помилок — 60% критеріїв

Рядок — істинний клас, стовпець — рішення моделі. Діагональ показує правильні прогнози.
У raw-матриці значення є кількостями; у normalized кожен рядок ділиться на свою суму,
тому діагональ є recall класу. У всіх raw-матриць одна шкала кольору, normalized — 0…100%.
Окремі {2 * n_pairs} зображень і відповідні CSV збережені в `results/figures` та `results/tables`.
"""),
        code('figure("confusion_raw_grid", 1400)'),
        code('figure("confusion_normalized_grid", 1400)'),
        code('key = selected["extractor"] + "_" + selected["classifier"]\n'
             'print("Обрана пара:", NAMES[selected["extractor"]], "+", NAMES[selected["classifier"]])\n'
             'display(table("confusion_" + key + ".csv", index_col=0))\n'
             'display(table("classification_report_" + key + ".csv", index_col=0).round(4))'),
        md(f"""## 4. Залежність accuracy від глибини заморожених шарів — 20% критеріїв

Одна початкова голова після {config['warmup_epochs']} warm-up епох; далі максимум {config['fine_tune_epochs']} епох з Adam {str(format(config['fine_tune_learning_rate'], '.0e')).replace('e-0', 'e-')}.
Режими: розморожено останні {policies} шарів MobileNetV2 (155 = усі шари).
Усі 52 BatchNorm frozen та inference. Режими використовують ті самі дані й перестановки train.
EarlyStopping відстежує val loss, patience={config['early_stopping_patience']}; вимірювання — **після відновлення мінімуму val loss**.

`unfreeze_last` — ширина кінцевого вікна шарів, а не число навчуваних параметризованих шарів.
На осі X — фактичне число frozen шарів у backbone, включно з BatchNorm та шарами без ваг.
Додатково наведено число навчуваних шарів із вагами та число trainable параметрів.

Для прискорення незмінний префікс обчислено один раз у float32; trainable суфікс донавчається.
Враховано residual-ребра через межу префікса. Числова еквівалентність перевірена до й після fit.
Аугментація не застосовувалася. Можна виконати повну CNN без кешу через конфігурацію.
"""),
        code('freeze = table("freeze_depth_validation.csv").sort_values("frozen_layers", ascending=False)\n'
             'display(freeze[["unfreeze_last", "frozen_layers", "trainable_layer_flags", '
             '"weighted_trainable_layers", "trainable_backbone_params", "epochs_run", '
             '"restored_epoch", "val_accuracy", "val_macro_f1"]].round(4).reset_index(drop=True))\n'
             'figure("freeze_depth_accuracy_trend")'),
        code('figure("freeze_learning_accuracy", 1300)\nfigure("freeze_learning_loss", 1300)'),
        code('display(Markdown(f"**Вибрана політика:** {policy}. "\n'
             '                 f"Test accuracy: **{depth[\'test\'][\'accuracy\']:.2%}**, "\n'
             '                 f"macro-F1: **{depth[\'test\'][\'macro_f1\']:.4f}**."))\n'
             'figure("freeze_best_confusion", 800)\nfigure("freeze_best_confusion_normalized", 800)'),
        md("""## 5. Характерні помилки — 20% критеріїв

Спочатку частоти: які направлені пари «істинний → прогноз» найчастіші в обраної моделі,
і чи повторюються ті самі клітинки в усіх класичних пар та в обраної CNN (тоді помилка систематична).
Галерея `typical_errors` показує до чотирьох фото для найчастіших пар,
`confident_errors` — найвпевненіші хибні рішення. Confidence є `predict_proba` саме класу,
повернутого `predict`; у SVM ці два правила не завжди мають один argmax.
Візуальні пояснення нижче є гіпотезами за переглянутими фото, а не доказом мотиву CNN.
"""),
        code('display(table("common_error_pairs.csv"))\nfigure("common_error_pairs")'),
        code('across = table("error_pairs_across_models.csv")\n'
             'display(across[["true_class", "pred_class", "mean_classical", "min_classical", "max_classical", '
             '"selected_cnn"]].head(8).round(2))\n'
             'figure("error_pairs_across_models")'),
        code('figure("typical_errors", 1300)\nfigure("confident_errors", 1300)'),
        code('observations = table("error_observations_ua.csv")\n'
             'display(observations.error_category.value_counts().rename("кількість фото"))\n'
             'with pd.option_context("display.max_colwidth", None):\n'
             '    display(observations[["image_id", "true_class", "pred_class", "error_category", '
             '"visible_observation", "possible_explanation"]])'),
        md("""## 6. Перевірки, висновки та межі результату

Висновок стосується одного датасету, одного seed і одного split. Це не доказ універсальної
переваги одного екстрактора чи алгоритму. Присутні складні фони, ймовірний шум міток і
майже-дублікати. Різницю у кілька правильних фото не трактуємо як доведену статистичну перевагу.
Моделі не навчають клас «не квітка», не оцінюють повний ботанічний асортимент і не мають
зовнішньої перевірки калібрування впевненості.
"""),
        code('audit = read_json("results/audit.json")\n'
             'print("Статус:", audit["status"], "| перевірок:", len(audit["checks"]))\n'
             'display(pd.DataFrame(audit["checks"])[["check", "passed"]])'),
        code('display(Markdown((ROOT / "report/CONCLUSIONS_UA.md").read_text(encoding="utf-8")))'),
        md("""## 7. План захисту на 8–10 хвилин

1. **Учасник 1:** дані, split, перевірка дублікатів, MobileNetV2 і VGG16, t-SNE — 2–3 хв.
2. **Учасник 2:** ResNet50, SVM, warm-up і контроль глибини заморожування, графік — 3 хв.
3. **Учасник 3:** RF, порівняння пар, матриці, частоти помилок, конкретні фото — 3–4 хв.

Повний текст виступу та запитання: `report/DEFENSE_UA.md`.
Відповідність умовам: `report/REQUIREMENTS_CHECKLIST.md`.
"""),
        md("""## Джерела

* Надана презентація `Labs.pdf`, слайд 14, Transfer learning:
  https://github.com/dryabokon/Docs/blob/master/ML/Labs.pdf
* TFDS TensorFlow Flowers: https://www.tensorflow.org/datasets/catalog/tf_flowers
* Код датасету і класи:
  https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/tf_flowers/tf_flowers_dataset_builder.py
* Keras Transfer learning: https://keras.io/guides/transfer_learning/
* MobileNetV2: https://keras.io/api/applications/mobilenet/mobilenet_models/
* ResNet: https://keras.io/api/applications/resnet/
* VGG16: https://keras.io/api/applications/vgg/
* Метрики: https://scikit-learn.org/stable/modules/model_evaluation.html
* t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

Атрибуція фото: `data/FLOWERS_LICENSE.txt`. Ваги ImageNet — Keras Applications.
"""),
    ]


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / "notebooks" / name)


def generate():
    config = read_config()
    n_pairs = len(EXTRACTORS) * len(CLASSIFIERS)
    n_figures = len(expected_figures(config))
    write("00_Results_and_Defense.ipynb", [
        md(f"""# Assignment 2 · Transfer Learning
## TensorFlow Flowers — готові результати й захист

Цей ноутбук відкриває **вже виконані експерименти**. Усі виводи збережено.
Можна натиснути `Run All`: він читає результати й показує рисунки, не навчає моделі
і не потребує TensorFlow. Для повторного навчання використайте ноутбук `01`.

| Критерій зі слайда 14 | Частка | Матеріали |
|---|---:|---|
| Confusion matrices | 60% | {n_pairs} raw, {n_pairs} normalized, матриці обраної CNN |
| Accuracy trend | 20% | {len(config['unfreeze_last'])} режимів заморожування, actual frozen layers, validation accuracy |
| Typical errors | 20% | частоти пар у всіх моделях, реальні фото, confidence, категорії причин |
"""), code(INITIAL), code(summary_code(n_figures)), *shared_results(config)
    ])
    research = [
        md("""# Assignment 2 · Transfer Learning із кодом
## MobileNetV2 / ResNet50 / VGG16 × Logistic Regression / SVM / Random Forest

Повний дослідницький маршрут для Jupyter. У корені архіву є модулі `src/` з усією реалізацією.
**Готові результати вже включені.** За замовчуванням `RETRAIN=False`; повне повторення
вмикається наступною змінною. Перший повторний запуск даних завантажить приблизно 219 MiB JPEG.
Встановлення залежностей описано в README. Рекомендовано Python 3.12.
"""),
        code(INITIAL),
        code('RETRAIN = False  # True: повторити підготовку, benchmark і всі режими CNN (години на CPU)\n'
             'RUN_CUSTOM_IMAGE_DEMO = True\n'
             'import sys, os, subprocess\n'
             'sys.path.insert(0, str(ROOT))\n'
             'from src.common import setup, read_config, runtime_metadata, save_json, EXTRACTORS\n'
             'config = read_config()\nsetup(config)\n'
             'print("Налаштування:")\ndisplay(config)'),
        md(r"""## Мета, гіпотези й метрики

Перевіряємо різницю між екстракторами, класифікаторами й політиками заморожування.
Transfer learning використовує ознаки, вивчені на ImageNet, у новій задачі на 5 класів.
Feature extraction залишає CNN незмінною; fine-tuning оновлює дозволений суфікс CNN.

$$Accuracy=\frac{N_{correct}}{N},\quad F1_k=\frac{2P_kR_k}{P_k+R_k},\quad
MacroF1=\frac{1}{5}\sum_{k=1}^5 F1_k.$$

Умови слайда 14 — benchmark, matrices, frozen-depth trend та errors; конкретні CNN,
датасет, epochs і split обрано в плані проєкту.
"""),
        md("""## A. Підготовка даних і фіксація split

TFDS надає назви міток; оригінальні JPEG беруться з того самого офіційного архіву TensorFlow,
який використовує TFDS. Цей маршрут зберігає первинні JPEG й атрибуцію, не змінює набір даних.
Split стратифікований на рівні SHA-256-груп, щоб точні дублікати не перетнули підвибірки.
"""),
        code('from src.data import prepare\n'
             'if RETRAIN:\n    prepare(config)\n    save_json(ROOT / "results/runtime.json", runtime_metadata(config))\n'
             'data_manifest = pd.read_csv(ROOT / "data/manifest.csv")\n'
             'assert len(data_manifest) == 3670\n'
             'assert data_manifest.groupby("sha256").split.nunique().max() == 1\n'
             'display(data_manifest.head())'),
        md("""## B. CNN-ознаки та порядок прикладів

Вхід — RGB float32 після resize до 224×224. MobileNetV2: масштабування до [-1, 1].
ResNet50 і VGG16: RGB → BGR і центрування ImageNet. `preprocess_input` свій для кожної мережі.
Не можна виконувати одну універсальну нормалізацію. Під час extraction train/val/test
не перемішуються; кожен рядок ознак має свій image_id.
"""),
        code('from src.features import extract, load_features\n'
             'shapes = []\n'
             'for name in EXTRACTORS:\n'
             '    if RETRAIN:\n        extract(config, name)\n'
             '    cached = load_features(config, name)\n'
             '    for split in ["train", "val", "test"]:\n'
             '        shapes.append({"extractor": name, "split": split, "shape": str(cached["X_"+split].shape)})\n'
             '    cached.close()\n'
             'display(pd.DataFrame(shapes))'),
        md(f"""## C. Навчання {n_pairs} кандидатів

У `src/benchmark.py` спочатку виконуються всі fit/validation, записується вибір за validation,
і тільки після цього обчислюються test-прогнози всіх фіксованих кандидатів.
Підбір гіперпараметрів на test відсутній. Probability SVM оцінюється внутрішньо лише з train.
"""),
        code('from src.benchmark import run as run_benchmark, classifiers\n'
             'import inspect\nfrom IPython.display import Code\n'
             'display(Code(inspect.getsource(classifiers), language="python"))\n'
             'if RETRAIN:\n    run_benchmark(config)\n'
             'metrics = table("benchmark_metrics.csv")\n'
             'selected = read_json("results/selected_benchmark.json")'),
        md("""## D. Контрольоване дослідження глибини заморожування

Після спільного warm-up кожен режим починається з тієї самої голови й ваг ImageNet.
Не продовжуємо режим 20 із результату режиму 0 і не продовжуємо 50 із результату 20.
Після зміни trainable створюється новий Adam та виконується compile.
Окремі процеси звільняють TensorFlow-памʼять між великими експериментами.
"""),
        code('from src.freeze import configure\n'
             'display(Code(inspect.getsource(configure), language="python"))\n'
             'if RETRAIN:\n'
             '    commands = [("warmup", [])] + [("freeze", ["--policy", str(k)]) for k in config["unfreeze_last"]]\n'
             '    commands += [("freeze-finalize", []), ("plots", []), ("audit", []), ("report", [])]\n'
             '    for stage, extra in commands:\n'
             '        subprocess.run([sys.executable, "-m", "src.pipeline", "--stage", stage, *extra],\n'
             '                       cwd=ROOT, check=True, env=dict(os.environ, PYTHONHASHSEED="42"))\n'
             'depth = read_json("results/selected_depth_test.json")'),
        code(summary_code(n_figures)),
        *shared_results(config),
        md("""## E. Реальний прогноз довільного фото

Можна змінити `DEMO_IMAGE` на шлях до власного JPEG/PNG. У початковому запуску беремо
включене демонстраційне фото з train; воно показує роботу інтерфейсу, а не незалежну якість.
Якість виміряна окремо на test. Очікувані класи: dandelion, daisy, tulips, sunflowers, roses.
"""),
        code('demo_rows = pd.read_csv(ROOT / "data/demo_images/attribution_manifest.csv")\n'
             'demo_id = demo_rows.query("split == \'train\'").iloc[0].image_id\n'
             'DEMO_IMAGE = ROOT / "data/demo_images" / (demo_id + ".jpg")\n'
             'if RUN_CUSTOM_IMAGE_DEMO:\n'
             '    from src.predict import predict_image\n'
             '    display(Image(filename=str(DEMO_IMAGE), width=350))\n'
             '    display(predict_image(DEMO_IMAGE))'),
    ]
    write("01_Transfer_Learning_Flowers.ipynb", research)
    print("Created two Jupyter notebooks")


if __name__ == "__main__":
    generate()
