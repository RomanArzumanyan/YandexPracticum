import torch
import torch.nn as nn
from tqdm import tqdm
import random

DEVICE = torch.device("cuda")


class LstmPredictor(nn.Module):
    def __init__(self, tokenizer, hidden_dim=128):
        super().__init__()

        vocab_size = tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim,
                           batch_first=True, bidirectional=False)
        out_dim = hidden_dim
        self.fc = nn.Linear(out_dim, vocab_size)
        self.to(DEVICE)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        position = x.size(1) - 1
        hidden_forward = out[:, position, :out.size(2)]
        linear_out = self.fc(hidden_forward)
        return linear_out


def evaluate(model, loader, criterion):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_output = model(x_batch)
            loss = criterion(x_output, y_batch)
            preds = torch.argmax(x_output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            sum_loss += loss

    avg_loss = sum_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(model, n_epochs, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        for x_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            x_output = model(x_batch)
            loss = criterion(x_output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")


def inference(model, val_loader, tokenizer):
    model.eval()
    bad_cases, good_cases = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch, y_batch
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            for i in range(len(y_batch)):
                input_tokens = tokenizer(
                ).convert_ids_to_tokens(x_batch[i].tolist())
                true_tok = tokenizer.convert_ids_to_tokens([
                    y_batch[i].item()])[0]
                pred_tok = tokenizer(
                ).convert_ids_to_tokens([preds[i].item()])[0]

                if preds[i] != y_batch[i]:
                    bad_cases.append((input_tokens, true_tok, pred_tok))
                else:
                    good_cases.append((input_tokens, true_tok, pred_tok))

    random.seed(42)
    bad_cases_sampled = random.sample(bad_cases, 5)
    good_cases_sampled = random.sample(good_cases, 5)

    print("\nSome incorrect predictions:")
    for context, true_tok, pred_tok in bad_cases_sampled:
        print(
            f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")

    print("\nSome correct predictions:")
    for context, true_tok, pred_tok in good_cases_sampled:
        if true_tok == pred_tok:
            print(
                f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")
