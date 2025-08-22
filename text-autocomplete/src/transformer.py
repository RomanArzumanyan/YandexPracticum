from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm


class DistilGPT2():
    def __init__(self):
        model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side='left')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            '[PAD]')
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
        )

    def autocomplete(self, prompt: str) -> str:
        ret = self.generator(
            prompt,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            max_length=80,
        )[0]

        prompt_len = len(prompt.split())
        words = ret["generated_text"].split()
        worlds_len = len(words)
        return words[prompt_len if worlds_len > prompt_len else -1]

    def inference(self, train_set: list[str]):
        correct = 0
        data = []
        preds = []

        for line in tqdm(train_set):
            words = line.split()
            context = ' '.join(words[0:-1])
            target = words[-1]
            data.append({
                'context': context,
                'target': target,
                'length': len(words[0:-1])
            })

        for ret in tqdm(self.generator(KeyDataset(Dataset.from_list(data), "context"), max_new_tokens=20, num_workers=2)):
            words = ret[0]["generated_text"].split()
            preds.append(words)

        for i in range(0, len(preds)):
            target, prompt_len = data[i]['target'], data[i]['length']
            pred_words = preds[i]
            if pred_words[min(prompt_len, len(pred_words) - 1)] == target:
                correct += 1

        accuracy = float(correct) / float(len(data))
        print(f"Accuracy: {accuracy:.2%}")
