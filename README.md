# Assignment 2: Transfer Learning — TensorFlow Flowers

Перевірка за наданим слайдом і перелік виправлень: `report/REVIEW_UA.md`.

Завершений дослідницький проєкт для Jupyter. Усі метрики, фотографії помилок і графіки в `results/` отримані реальним запуском TensorFlow та scikit-learn, а не створені як демонстраційні числа.

## З чого почати

1. Розпакуйте архів цілком. Не переносіть окремий навчальний ноутбук без папок `src`, `configs`, `data`, `results` і `models`.
2. Для швидкого перегляду відкрийте **`report/Results_UA.html`** у браузері: інтернет, Jupyter і TensorFlow не потрібні. Також є `report/Results_UA.pdf`.
3. Для показу у Jupyter відкрийте **`notebooks/00_Results_and_Defense.ipynb`**. Виводи вже збережені. `Run All` цього ноутбука лише читає готові результати.
4. Для вивчення коду, власних прогнозів та повторного дослідження відкрийте **`notebooks/01_Transfer_Learning_Flowers.ipynb`**. За замовчуванням він використовує перевірені кеші; `RETRAIN=True` запускає навчання.

## Запуск у Windows / Jupyter

Перевірене середовище: Python 3.12, TensorFlow CPU 2.20.0, Keras 3.15.1, scikit-learn 1.8.0. Повний перелік фактичних версій і параметрів є в `results/runtime.json`. Рекомендовано Python 3.12; для цих зафіксованих залежностей потрібен Python 3.11 або новіший.

```powershell
cd assignment2_flowers
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name flowers --display-name "Python (Flowers)"
python -m jupyter lab
```

Виберіть ядро **Python (Flowers)**. Якщо PowerShell блокує активацію, використовуйте напряму `.venv\Scripts\python.exe` замість `python`; змінювати системну політику не потрібно.

### Linux / macOS

```bash
cd assignment2_flowers
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name flowers --display-name "Python (Flowers)"
python -m jupyter lab
```

### Лише перегляд результатів

Для `00_Results_and_Defense.ipynb` достатньо `requirements-viewer.txt`; TensorFlow не потрібен. Збережені виводи ноутбука, HTML і PDF можна переглядати без встановлення залежностей.

### Google Colab

1. Увімкніть GPU за наявності та завантажте ZIP у `/content`.
2. Розпакуйте його командою `!unzip -q /content/Assignment2_Flowers_Jupyter.zip -d /content`.
3. Відкрийте `01_Transfer_Learning_Flowers.ipynb` і задайте робочий каталог `%cd /content/assignment2_flowers` перед імпортами.
4. Для використання саме перевіреного набору залежностей виконайте `%pip install -r requirements.txt`, після зміни TensorFlow перезапустіть сеанс.
5. Результати зберігайте у Google Drive або завантажте всю папку `results/`, `models/` та виконаний ноутбук. CPU також підтримується.

## Повторне виконання

У корені проєкту:

```bash
python -m src.pipeline --stage all
```

Порядок окремих етапів:

```bash
python -m src.pipeline --stage prepare
python -m src.pipeline --stage extract --model mobilenetv2
python -m src.pipeline --stage extract --model resnet50
python -m src.pipeline --stage extract --model vgg16
python -m src.pipeline --stage benchmark
python -m src.pipeline --stage warmup
# по одному запуску на кожен режим із configs/full.json: 0 10 20 35 50 75 100 115 155
python -m src.pipeline --stage freeze --policy 0
python -m src.pipeline --stage freeze --policy 155
python -m src.pipeline --stage freeze-finalize
python -m src.pipeline --stage plots
python -m src.pipeline --stage audit
```

`all` повторно навчає класифікатори й усі дев'ять режимів CNN. Готові ознаки повторно використовуються лише за збігу відбитка manifest, розміру зображень, моделі та TensorFlow. Для незалежного нового експерименту створіть копію папки. Не видаляйте чи змінюйте спільний `manifest.csv` у команді.

Оригінальні 3 670 JPEG не дублюються в основному архіві: перший `prepare` завантажить офіційний архів розміром приблизно 219 MiB. Порядок назв і класів зафіксований, SHA-256 архіву перевіряється. Демонстраційні фото та їх атрибуція включені в `data/demo_images`. Ваги MobileNetV2 і ResNet50, усі навчені класифікатори, дев'ять CNN-checkpoint і кеші ознак трьох CNN включені, тому готові результати та прогноз власного фото доступні без завантаження моделей. Ваги VGG16 (≈ 56 MiB) не дублюються: вони потрібні лише для повторного витягування ознак і завантажуються Keras автоматично.

На диску для повного повторення потрібно орієнтовно 5–7 GB вільного місця: найбільше займає тимчасовий кеш активацій поточного режиму (до ≈ 2.2 GB) і ваги/чекпойнти. RAM 8 GB достатньо в перевіреному CPU-запуску; кеш читається порціями з диска. Час залежить від обладнання: на 2-ядерному CPU витягування ознак VGG16 тривало ≈ 9 хв, повне розморожування MobileNetV2 ≈ 13 хв; фактичний час етапів записаний у CSV/JSON.

## Хронологія запусків

Основний запуск виконано 31.10.2026 (`results/runtime.json`). 01.11.2026 дослідження розширено: додано VGG16 і пʼять нових режимів заморожування (10, 35, 75, 115, 155) з тими самими версіями пакетів, але на іншому CPU (`results/runtime_extension.json`). Перед розширенням режим last 20 повторно навчено на новій машині: validation accuracy і macro-F1 збіглися до останнього знака (`results/reproducibility_check.json`), тож старі й нові точки тренду порівнянні. Початкові шість класичних пар перенавчено з тих самих ознак — їхні рішення й метрики не змінилися (ймовірності відрізняються лише на рівні 1e-12). Усі 3670 JPEG перевірено за SHA-256 із `data/manifest.csv`.

## Що досліджено

| Частина | Реалізація |
|---|---|
| Дані | TensorFlow Flowers: 3670 фото, 5 класів; 2569 train / 550 val / 551 test |
| Екстрактори | MobileNetV2 (1280 ознак), ResNet50 (2048 ознак), VGG16 (512 ознак), pretrained ImageNet |
| Класифікатори | Logistic Regression + StandardScaler; SVM RBF + StandardScaler; Random Forest |
| Benchmark | Усі 9 пар (3 CNN × 3 класифікатори), accuracy / macro-F1 / weighted-F1, per-class звіти |
| Заморожування | 9 режимів: розморожено останні 0 / 10 / 20 / 35 / 50 / 75 / 100 / 115 / усі 155 шарів MobileNetV2, BatchNorm завжди frozen |
| Матриці | 9 raw + 9 normalized, дві оглядові сітки 3×3; дві матриці вибраної CNN |
| Помилки | Частоти пар, ті самі пари в усіх моделях, характерні фото, галерея впевнених помилок, категорії причин |
| Ознаки | t-SNE тестових ознак трьох заморожених CNN і донавченої MobileNetV2 (як на слайді 14) |
| Відтворюваність | Seed 42, manifest SHA-256, runtime, логи, збережені прогнози й моделі |

### Важливе уточнення до початкового плану

У 3670 записах є лише 3667 унікальних SHA-256: три пари точних дублікатів. Одна пара має суперечливі офіційні класи `roses` / `tulips`. Усі записи й мітки збережені; split створено на рівні SHA-256-груп, тому однакове фото не перетинає train/val/test. Групова стратифікація зберігає наближені 70/15/15 і дала рівно 2569/550/551 записів. Аудит знаходиться в `results/tables/duplicate_audit.csv`. Це свідоме покращення початкового плану. Майже однакові, але побайтово різні фото окремо не усувалися.

## Як запобігли витоку даних

Scaler і класифікатори навчаються тільки на train. Краща класична пара визначена за validation macro-F1; вибір записано **до** test-прогнозів. Test-показники дев'яти заздалегідь заданих моделей є порівняльним підсумком, а не критерієм перевибору переможця. Режим CNN вибрано за validation accuracy відновленого checkpoint, потім оцінено на test тільки його. Після EarlyStopping відновлюється мінімум **val loss**; це не обов'язково епоха максимуму val accuracy.

Для кожної політики: однакова pretrained CNN, одна голова після 3 warm-up епох, однакові перестановки train, Adam 1e-5, максимум 6 епох і patience 2. BatchNorm frozen, backbone викликається з `training=False`. Зміна `trainable` супроводжується новою компіляцією.

Незмінний префікс CNN обчислюється один раз і кешується у float32 (для режиму «усі шари» кешується сам підготовлений вхід). У residual-блоках зберігаються **всі** ребра через межу розрізу. Числова еквівалентність із повною CNN перевіряється до й після навчання. Суфікс справді донавчається. Це прискорення не є заміною fine-tuning на готові незмінні ознаки. Його можна відключити в `configs/full.json`, задавши `cache_frozen_prefix=false`. Аугментація не застосовується.

## Прогноз нового фото

```bash
python -m src.predict --image path/to/my_flower.jpg
```

Модель вибирає один із п'яти відомих класів; окремий клас «не квітка» не навчався. `confidence` — оцінка ймовірності, а не гарантія. Для SVC рішення `predict` може відрізнятися від `argmax(predict_proba)`; у таблицях збережено справжнє рішення і ймовірність саме цього класу.

## Матеріали для здачі

* `notebooks/00_Results_and_Defense.ipynb` — показ готового дослідження.
* `notebooks/01_Transfer_Learning_Flowers.ipynb` — методика, код і повторення.
* `report/Results_UA.pdf`, `report/Results_UA.html`, `report/RESULTS_UA.md` — фактичний звіт.
* `report/DEFENSE_UA.md` — виступ трьох учасників і відповіді на запитання.
* `report/REQUIREMENTS_CHECKLIST.md` — відповідність критеріям 60/20/20.
* `results/figures/` — усі 36 PNG у високій роздільності.
* `results/tables/` — метрики, матриці, історії навчання й аудит.
* `results/predictions/` — рішення для кожного test-фото.

## Джерела

1. Надана викладацька презентація `Labs.pdf`, слайд 14, Transfer learning. [Оригінал](https://github.com/dryabokon/Docs/blob/master/ML/Labs.pdf).
2. [TFDS tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers), [код і порядок класів](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/tf_flowers/tf_flowers_dataset_builder.py).
3. [Keras Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).
4. [MobileNetV2](https://keras.io/api/applications/mobilenet/mobilenet_models/), [ResNet](https://keras.io/api/applications/resnet/), [VGG16](https://keras.io/api/applications/vgg/).
5. [scikit-learn: оцінювання класифікаторів](https://scikit-learn.org/stable/modules/model_evaluation.html).

Права на оригінальні фотографії й атрибуція наведені у `data/FLOWERS_LICENSE.txt`; це не власні фотографії авторів проєкту. Вимоги 60/20/20 походять із наданого слайда. Датасет, моделі, split і протокол — дослідницькі рішення цього проєкту.
