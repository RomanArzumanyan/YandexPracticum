import torch
import torch.nn as nn
from tqdm import tqdm
import random
import evaluate
from torch.nn.utils.rnn import pack_padded_sequence


DEVICE = torch.device("cuda")
HIDDEN_DIM = 128
ROUGE = evaluate.load("rouge")


class Lstm(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()

        vocab_size = tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, HIDDEN_DIM)
        self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        out_dim = HIDDEN_DIM
        self.fc = nn.Linear(out_dim, vocab_size)
        self.to(DEVICE)

    def forward(self, contexts, lengths):
        emb = self.embedding(contexts)
        packed_emb = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed_emb)
        out = self.fc(hidden[-1]).squeeze(0)
        return out


def evaluate_(model, loader, criterion, tokenizer, calc_rouge: bool = False):
    predictions = []
    references = []

    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch['contexts']
            lengths = batch['lengths']
            tokens = torch.tensor(batch['tokens']).to(DEVICE)
            pred_tokens = model(inputs, lengths)
            loss = criterion(pred_tokens, tokens)
            preds = torch.argmax(pred_tokens, dim=1)
            correct += (preds == tokens).sum().item()
            total += tokens.size(0)
            sum_loss += loss
            predictions.append(tokenizer.decode(
                preds, skip_special_tokens=True))
            references.append(tokenizer.decode(
                tokens, skip_special_tokens=True))

    avg_loss = sum_loss / len(loader)
    accuracy = correct / total
    rouge_score = ROUGE.compute(
        predictions=predictions, references=references) if calc_rouge else None
    return avg_loss, accuracy, rouge_score


def train(model, n_epochs, l_rate, tokenizer, train_loader, val_loader) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.
        for batch in tqdm(train_loader):
            inputs = batch['contexts']
            lengths = batch['lengths']
            tokens = torch.tensor(batch['tokens']).to(DEVICE)
            optimizer.zero_grad()
            pred_tokens = model(inputs, lengths)
            loss = criterion(pred_tokens, tokens)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc, val_rouge = evaluate_(
            model, val_loader, criterion, tokenizer, True)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")

        if (val_rouge):
            print('ROUGE metrics')
            for k, v in val_rouge.items():
                print(f"{k}: {v:.4f}")


def inference(model, loader, tokenizer, interactive=False) -> list[str]:
    model.eval()
    bad_cases, good_cases = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch['contexts']
            lengths = batch['lengths']
            tokens = batch['tokens']
            logits = model(inputs, lengths)
            preds = torch.argmax(logits, dim=1)
            for i in range(len(tokens)):
                input_tokens = tokenizer.decode(
                    inputs[i].tolist(), skip_special_tokens=True)
                true_tok = tokenizer.decode([tokens[i].item()])
                pred_tok = tokenizer.decode([preds[i].item()])

                if not interactive:
                    if preds[i] != tokens[i]:
                        bad_cases.append((input_tokens, true_tok, pred_tok))
                    else:
                        good_cases.append((input_tokens, true_tok, pred_tok))
                else:
                    good_cases.append(pred_tok)

    if not interactive:
        random.seed(42)
        bad_cases_sampled = random.sample(bad_cases, 5)
        good_cases_sampled = random.sample(good_cases, 5)

        print("\nSome incorrect predictions:")
        for context, true_tok, pred_tok in bad_cases_sampled:
            print(
                f"Input: {context} | True: {true_tok} | Predicted: {pred_tok}")

        print("\nSome correct predictions:")
        for context, true_tok, pred_tok in good_cases_sampled:
            if true_tok == pred_tok:
                print(
                    f"Input: {context} | True: {true_tok} | Predicted: {pred_tok}")
    else:
        return good_cases

    return []
