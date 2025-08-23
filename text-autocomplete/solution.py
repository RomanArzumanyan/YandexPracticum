from src import data_utils, data_set, lstm_model, transformer

TOKENIZER = data_set.TOKENIZER
MIN_LEN = data_set.MIN_LEN
ROUGE = lstm_model.ROUGE


url = "https://code.s3.yandex.net/deep-learning/tweets.txt"
filename = "data/raw_dataset.txt"
data_utils.download_from_url(url, filename)


# Аргумент cap ограничивает число строк, которые будут загружены.
# Значения до 100К удобны для проверки. Тетрадка отработает за несколько минут.
clean = data_utils.clean_up(data_utils.load_dataset(
    "data/raw_dataset.txt", cap=10000))

splits = data_utils.split_dataset(clean)

data_sets = []
data_loaders = []
for split in [splits['train'], splits['val']]:
    dset, dloader = data_set.prepare_data(split)
    data_sets.append(dset)
    data_loaders.append(dloader)


lstm = lstm_model.Lstm(TOKENIZER)
lstm.train_model(n_epochs=3, l_rate=0.002, tokenizer=TOKENIZER,
                 train_loader=data_loaders[0], val_loader=data_loaders[1])


NUM_WORDS = 3


# Заполняем датасет для инференса, т. к. мы обучаем модель предсказывать только
# 1 слово, а инференс хотим делать для нескольких.
lstm_data = []
true_data = []
for i in range(0, len(splits['test'])):
    words = splits['test'][i].split()
    if len(words) < MIN_LEN + NUM_WORDS:
        continue
    ctx_len = len(words) - NUM_WORDS
    context = ' '.join(words[:ctx_len])
    target = ' '.join(words[ctx_len:])
    lstm_data.append(context)
    true_data.append(context + ' ' + target)

# Данные для инференса трансформера будут точно такие же, делаем копию.
gpt2_data = lstm_data


# Проходимся по всему датасету и генерируем массив предсказанных слов.
# Далее, прибавляем предсказанное слово ко входам.
for i in range(0, NUM_WORDS):
    _, dloader = data_set.prepare_data(
        lstm_data, shuffle=False, num_targets=0)
    preds = lstm.inference(loader=dloader, tokenizer=TOKENIZER)
    lstm_data = [x + ' ' + y for x, y in zip(lstm_data, preds)]

rouge_score = ROUGE.compute(predictions=lstm_data, references=true_data)
print('ROUGE metrics')
for k, v in rouge_score.items():
    print(f"{k}: {v:.4f}")


gpt2 = transformer.DistilGPT2()
for i in range(0, NUM_WORDS):
    preds = gpt2.inference(gpt2_data)
    gpt2_data = [x + ' ' + y for x, y in zip(gpt2_data, preds)]

rouge_score = ROUGE.compute(predictions=gpt2_data, references=true_data)
print('ROUGE metrics')
for k, v in rouge_score.items():
    print(f"{k}: {v:.4f}")


print("Some LSTM predictions:")
for i in range(0, 3):
    print(lstm_data[i])

print("Some GPT2 predictions:")
for i in range(0, 3):
    print(gpt2_data[i])

print("Actual sentences:")
for i in range(0, 3):
    print(true_data[i])
