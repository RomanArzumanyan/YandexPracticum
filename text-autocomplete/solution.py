from src import data_utils, next_token_dataset, lstm_model


TOKENIZER = next_token_dataset.TOKENIZER

print(f"Load dataset \n")
dataset = data_utils.load_dataset("data/raw_dataset.txt", cap=-1)

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
lstm_model.train(model, n_epochs=3, l_rate=0.002, tokenizer=TOKENIZER,
                 train_loader=data_loaders[0], val_loader=data_loaders[1])

print(f"Inference \n")
lstm_model.inference(
    model, loader=data_loaders[2], tokenizer=TOKENIZER)

print(f"Autocomplete \n")
while True:
    user_input = input("Enter sentence, \"quit\" to exit: ")
    if user_input == "quit":
        break

    clean = data_utils.clean_up([user_input])
    _, dloader = next_token_dataset.prepare_data(clean)
    print(lstm_model.inference(
        model, loader=dloader, tokenizer=TOKENIZER, interactive=True))

print(f"Autoregression\n")
num_tokens = int(input("Enter number of tokens to predict: "))
while True:
    user_input = input("Enter sentence, \"quit\" to exit: ")
    if user_input == "quit":
        break

    for i in range(0, num_tokens):
        clean = data_utils.clean_up([user_input])
        _, dloader = next_token_dataset.prepare_data(clean)
        user_input = user_input + " " + \
            lstm_model.inference(model, loader=dloader,
                                 tokenizer=TOKENIZER, interactive=True)[-1]
    print(user_input)
