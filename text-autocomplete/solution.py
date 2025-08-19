from src import data_utils, next_token_dataset, lstm_model


TOKENIZER = next_token_dataset.TOKENIZER

print(f"Load dataset \n")
dataset = data_utils.load_dataset("data/raw_dataset.txt", cap=10000)

print(f"Clean dataset \n")
clean = data_utils.clean_up(dataset)

print(f"Split dataset \n")
splits = data_utils.split_dataset(clean)

print(f"Prepare torch datasets \n")
data_sets = []
data_loaders = []
for split_name in splits:
    shuffle = 'test' == split_name
    dset, dloader = next_token_dataset.prepare_data(
        splits[split_name], shuffle)
    data_sets.append(dset)
    data_loaders.append(dloader)

print(f"Train \n")
model = lstm_model.LstmPredictor(TOKENIZER)
lstm_model.train(model, n_epochs=3, l_rate=0.003, tokenizer=TOKENIZER,
                 train_loader=data_loaders[0], val_loader=data_loaders[1])

print(f"Inference \n")
lstm_model.inference(
    model, loader=data_loaders[2], tokenizer=TOKENIZER)
