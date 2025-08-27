# %% [markdown]
# Импортируем модули и глобальные переменные на уровне модулей.
# 

# %%
from src import data_utils, data_set, lstm_model, transformer

TOKENIZER = data_set.TOKENIZER
MIN_LEN = data_set.MIN_LEN
ROUGE = lstm_model.ROUGE

# %% [markdown]
# Готовим данные:
# - Загружаем датасет.
# - Чистим его с помощью регулярных выражений.
# - Разбиваем на сеты для обучения и валидации.

# %%
url = "https://code.s3.yandex.net/deep-learning/tweets.txt"
filename = "data/raw_dataset.txt"
data_utils.download_from_url(url, filename)

# %%
# Аргумент cap ограничивает число строк, которые будут загружены.
# Значения до 100К удобны для проверки. Тетрадка отработает за несколько минут.
clean = data_utils.clean_up(data_utils.load_dataset(
    "data/raw_dataset.txt", cap=10000))

splits = data_utils.split_dataset(clean)

data_sets = []
data_loaders = []
for split in [splits['train'], splits['val']]:
    dset = data_set.TwitterDataset(split)
    data_sets.append(dset)
    data_loaders.append(dset.get_loader())

# %% [markdown]
# Обучаем LSTM модель

# %%
lstm = lstm_model.Lstm(TOKENIZER)
lstm.train_model(n_epochs=3, l_rate=0.002, tokenizer=TOKENIZER,
                 train_loader=data_loaders[0], val_loader=data_loaders[1])

# Сохраняем модель для инференса
lstm.save_state_dict("models/lstm_state_dict.pth")

# И целиком
lstm.save("models/lstm_entire.pth")

# %% [markdown]
# В ходе обучения предсказываем 3 слова для каждого предложения.
# 
# Почему так ? В задании сказано "предсказать оставшуюся 1/4 часть предложения". \
# Т. к. предложения разной длины, то элементы батча будут обработаны разное число раз. \
# Это ограничит возможности использования батчей, вот возможные варианты:
# - Отказаться от батчей вовсе.
# - Отсортировать предложения по длине и использовать маленький батч (чтобы длины в батче совпадали)
# - Зафиксирваоть число слов (1/4 минимальной длины предложения или максмальной и т. п.)
# 
# 
# Получается логика обработки, которая не имеет никакого отношения к обучению нейросетям. Как будто, эта работа лишняя. \
# Если вы настаиваете, я решу задачу одним из описанных способов.

# %%
NUM_WORDS = 3

# %% [markdown]
# Инференс с авторегрессией

# %%
# Заполняем датасет для инференса, т. к. мы обучаем модель предсказывать только
# 1 слово, а инференс хотим делать для нескольких.
lstm_data = []
true_preds = []
full_texts = []

for i in range(0, len(splits['test'])):
    words = splits['test'][i].split()
    if len(words) < MIN_LEN + NUM_WORDS:
        continue
    ctx_len = len(words) - NUM_WORDS
    context = ' '.join(words[:ctx_len])
    target = ' '.join(words[ctx_len:])
    lstm_data.append(context)
    true_preds.append(target)
    full_texts.append(context + " " + target)

# Данные для инференса трансформера будут точно такие же, делаем копию.
gpt2_data = lstm_data

# %%
# Здесь храним только предсказанные слова.
pred_data = [""] * len(lstm_data)

# Проходимся по всему датасету и генерируем массив предсказанных слов.
# Далее, прибавляем предсказанное слово ко входам.
for i in range(0, NUM_WORDS):
    dataset = data_set.TwitterDataset(lstm_data, shuffle=False, num_targets=0)
    preds = lstm.inference(loader=dataset.get_loader(), tokenizer=TOKENIZER)
    lstm_data = [x + ' ' + y for x, y in zip(lstm_data, preds)]
    pred_data = [x + ' ' + y for x, y in zip(pred_data, preds)]

rouge_score = ROUGE.compute(predictions=pred_data, references=true_preds)
print('ROUGE metrics')
for k, v in rouge_score.items():
    print(f"{k}: {v:.4f}")

# %% [markdown]
# Аналогичная процедура для трансформера.

# %%
pred_data = [""] * len(lstm_data)

gpt2 = transformer.DistilGPT2()
for i in range(0, NUM_WORDS):
    preds = gpt2.inference(gpt2_data)
    gpt2_data = [x + ' ' + y for x, y in zip(gpt2_data, preds)]
    pred_data = [x + ' ' + y for x, y in zip(pred_data, preds)]

rouge_score = ROUGE.compute(predictions=pred_data, references=true_preds)
print('ROUGE metrics')
for k, v in rouge_score.items():
    print(f"{k}: {v:.4f}")

# %% [markdown]
# Посмотрим на некоторые результаты работы

# %%
print("\nSome LSTM predictions:")
for i in range(0, 3):
    print(lstm_data[i])

print("\nSome GPT2 predictions:")
for i in range(0, 3):
    print(gpt2_data[i])

print("\nActual sentences:")
for i in range(0, 3):
    print(full_texts[i])

# %% [markdown]
# Выводы:
# - Использование трансформера избыточно для этой задачи.
# - LSTM с небольшим скрытым состоянием (128) справляется с задачей предсказания следующего слова лучше.
# - Эксперименты по увеличению размера скрытого состояния LSTM (до 384) показали, что на коротких текстах это бессмысленно. Accuracy достигает значения, близкого к максимальному уже на 1й эпохе, а потом прекращает рост.
# - Меньший размер скрытого состояния (128) даёт схожее значение accuracy уже на 3й эпохе обучения.
# - Траснформер выдаёт более "осмысленные", но менее точные предсказания при авторегрессии.


