from src import data_utils, next_token_dataset, lstm_model


TOKENIZER = next_token_dataset.tokenizer()

# Prepare dataset
try:
    dataset = data_utils.load_dataset("data/raw_dataset.txt", 70000)
    clean = list(map(data_utils.clean_up, dataset))
    split = data_utils.split_dataset(clean)
    train_set, train_loader = next_token_dataset.prepare_data(split[0])
    val_set, val_loader = next_token_dataset.prepare_data(split[1])
    model = lstm_model.LstmPredictor(TOKENIZER)
    lstm_model.train(model, 3, train_loader, val_loader)

except Exception as e:
    print(e)
    exit(1)
