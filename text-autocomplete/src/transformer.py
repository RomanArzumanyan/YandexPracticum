from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

# Max sentence length in tokens
MAX_LEN = 80
BATCH_SIZE = 256


class DistilGPT2():
    """
    Class with distilgpt2 text generation pipeline
    """

    def __init__(self):
        model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, truncation=True, padding_side="left",
            padding="max_length", max_length=MAX_LEN)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
            batch_size=BATCH_SIZE
        )

    def inference(self, train_set: list[str]) -> list[str]:
        """
        Predict 1 word for every given sentence

        Args:
            train_set (list[str]): list of prompts to autocomplete

        Returns:
            list[str]: list of prompts + predictions
        """
        data = []
        preds = []
        clean = []

        for line in train_set:
            data.append({
                'context': line,
                'length': len(line.split())
            })

        for ret in tqdm(self.generator(KeyDataset(Dataset.from_list(data), "context"), max_new_tokens=20)):
            preds.append(ret[0]["generated_text"].split())

        for i in range(0, len(preds)):
            prompt_len = data[i]['length']
            words = preds[i]
            if prompt_len >= len(words):
                # pipeline didn't autocomplete, that sometimes happen
                clean.append('zzz')
                continue
            clean.append(words[prompt_len])

        return clean
