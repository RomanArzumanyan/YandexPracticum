from src import data_utils, data_set, lstm_model, transformer
from tqdm import tqdm

TOKENIZER = data_set.TOKENIZER
MIN_LEN = data_set.MIN_LEN
NUM_TOK = 3
ROUGE = lstm_model.ROUGE

clean = data_utils.clean_up(data_utils.load_dataset(
    "data/raw_dataset.txt", cap=5000))

splits = data_utils.split_dataset(clean)

# Create dataset and data loader for train and validation
data_sets = []
data_loaders = []
for split in [splits['train'], splits['val']]:
    dset, dloader = data_set.prepare_data(split)
    data_sets.append(dset)
    data_loaders.append(dloader)

lstm = lstm_model.Lstm(TOKENIZER)
lstm_model.train(lstm, n_epochs=3, l_rate=0.002, tokenizer=TOKENIZER,
                 train_loader=data_loaders[0], val_loader=data_loaders[1])

# Fill inference data.
# Last NUM_TOK words from every sentence will be trimmed.
# During the inference, NN will predict NUM_TOK tokens for every sentence.
# This number is fixed because otherwise (tell NN to predict 1/4 of input) every
# dataset entry will undergo inference variable amount of times.
lstm_split = []
true_split = []
for i in range(0, len(splits['test'])):
    words = splits['test'][i].split()
    if len(words) < MIN_LEN + NUM_TOK:
        continue
    ctx_len = len(words) - NUM_TOK
    context = ' '.join(words[:ctx_len])
    target = ' '.join(words[ctx_len:])
    lstm_split.append(context)
    true_split.append(context + ' ' + target)

# Save ourselves some work in the near future.
gpt2_split = lstm_split

# Autoregression is done via concatenation of user input and model output
# Tell dataset not to trim last token because it's already done above.
for i in range(0, NUM_TOK):
    _, dloader = data_set.prepare_data(lstm_split, shuffle=False, num_targets=0)
    preds = lstm_model.inference(lstm, loader=dloader, tokenizer=TOKENIZER)
    lstm_split = [x + ' ' + y for x, y in zip(lstm_split, preds)]

rouge_score = ROUGE.compute(predictions=lstm_split, references=true_split)
print('ROUGE metrics')
for k, v in rouge_score.items():
    print(f"{k}: {v:.4f}")

gpt2 = transformer.DistilGPT2()
for i in range(0, NUM_TOK):
    preds = gpt2.inference(gpt2_split)
    gpt2_split = [x + ' ' + y for x, y in zip(gpt2_split, preds)]

rouge_score = ROUGE.compute(predictions=gpt2_split, references=true_split)
print('ROUGE metrics')
for k, v in rouge_score.items():
    print(f"{k}: {v:.4f}")
