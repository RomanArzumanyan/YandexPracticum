from src import data_utils, data_set, models


TOKENIZER = data_set.TOKENIZER

print(f"Load dataset \n")
dataset = data_utils.load_dataset("data/raw_dataset.txt", cap=100000)

print(f"Clean dataset \n")
clean = data_utils.clean_up(dataset)

print(f"Split dataset \n")
splits = data_utils.split_dataset(clean)

print(f"Prepare torch datasets \n")
data_sets = []
data_loaders = []
for split_name in splits:
    shuffle = 'test' == split_name
    dset, dloader = data_set.prepare_data(
        splits[split_name], shuffle)
    data_sets.append(dset)
    data_loaders.append(dloader)

print(f"Train \n")
lstm = models.LstmPredictor(TOKENIZER)
models.train(lstm, n_epochs=3, l_rate=0.002, tokenizer=TOKENIZER,
             train_loader=data_loaders[0], val_loader=data_loaders[1])

print(f"Inference \n")
models.inference(
    lstm, loader=data_loaders[2], tokenizer=TOKENIZER)

print(f"Autoregression. Model will predict last 25% of text. \n")
while True:
    user_input = input("Enter sentence, \"quit\" to exit: ")
    if user_input == "quit":
        break

    num_tokens = max(1, len(user_input.split()) // 3)

    for i in range(0, num_tokens):
        clean = data_utils.clean_up([user_input])
        _, dloader = data_set.prepare_data(clean)
        user_input = user_input + " " + \
            models.inference(lstm, loader=dloader,
                             tokenizer=TOKENIZER, interactive=True)[-1]
    print(user_input)

print(f"Inference with DistilGPT2 \n")

transformer = models.DistilGPT2()
transformer.inference(splits["test"][0:50])

while True:
    user_input = input("Enter sentence, \"quit\" to exit: ")
    if user_input == "quit":
        break

    num_tokens = max(1, len(user_input.split()) // 3)

    for i in range(0, num_tokens):
        clean = data_utils.clean_up([user_input])
        user_input = user_input + " " + transformer.autocomplete(user_input)
    print(user_input)
