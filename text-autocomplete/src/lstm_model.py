import torch
import torch.nn as nn
from tqdm import tqdm
import random
import evaluate
from torch.nn.utils.rnn import pack_padded_sequence


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HIDDEN_DIM = 128
ROUGE = evaluate.load("rouge")


class Lstm(nn.Module):
    """
    Single directional LSTM model class consists of following layers:
     - Embedding
     - LSTM
     - Linear
    """

    def __init__(self, tokenizer):
        super().__init__()

        vocab_size = tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, HIDDEN_DIM)
        self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        out_dim = HIDDEN_DIM
        self.fc = nn.Linear(out_dim, vocab_size)
        self.to(DEVICE)

    def forward(self, contexts, lengths):
        """
        Forward method which predicts single token

        Args:
            contexts : batch with model inputs
            lengths : input lengths

        Returns:
            _type_: tensor, first dimension is batch size, second is HIDDEN_DIM, consists of logits
        """
        emb = self.embedding(contexts)
        packed_emb = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed_emb)
        out = self.fc(hidden[-1]).squeeze(0)
        return out

    def evaluate_(self, loader, criterion, tokenizer, calc_rouge: bool = False) -> tuple:
        """
        Run evaluation

        Args:
            loader : torch data loader
            criterion : loss function
            tokenizer : tokenizer
            calc_rouge (bool, optional): calc ROUGE metrics. Defaults to False.

        Returns:
            tuple: tuple with avg_loss, accuracy and ROUGE score
        """
        predictions = []
        references = []

        self.eval()
        correct, total = 0, 0
        sum_loss = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch['contexts']
                lengths = batch['lengths']
                tokens = torch.tensor(batch['tokens']).to(DEVICE)
                pred_tokens = self(inputs, lengths)
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

    def train_model(self, n_epochs, l_rate, tokenizer, train_loader, val_loader) -> None:
        """
        Train model

        Args:
            n_epochs : number of epochs
            l_rate : learning rate
            tokenizer : tokenizer
            train_loader : train data loader
            val_loader : validation data loader
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=l_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0.
            for batch in tqdm(train_loader):
                inputs = batch['contexts']
                lengths = batch['lengths']
                tokens = torch.tensor(batch['tokens']).to(DEVICE)
                optimizer.zero_grad()
                pred_tokens = self(inputs, lengths)
                loss = criterion(pred_tokens, tokens)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss, val_acc, val_rouge = self.evaluate_(
                val_loader, criterion, tokenizer, True)
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")

            if (val_rouge):
                print('ROUGE metrics')
                for k, v in val_rouge.items():
                    print(f"{k}: {v:.4f}")

    def inference(self, loader, tokenizer) -> list[str]:
        """
        Run inference

        Args:
            loader : test data loader
            tokenizer : tokenizer

        Returns:
            list[str]: list of predicted words
        """
        self.eval()
        ret = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch['contexts']
                lengths = batch['lengths']
                logits = self(inputs, lengths)
                preds = torch.argmax(logits, dim=1)
                for i in range(len(inputs)):
                    pred_tok = tokenizer.decode([preds[i].item()])
                    ret.append(pred_tok)
        return ret

    def save_state_dict(self, path: str) -> None:
        """
        Save model state

        Args:
            path (str): path to output
        """
        torch.save(self.state_dict(), path)

    def save(self, path: str) -> None:
        """
        Save entire model

        Args:
            path (str): path to output
        """
        torch.save(self, path)
