from transformers import (
    AutoModelWithLMHead,
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    GPTNeoForCausalLM,
    GPT2Tokenizer,
)
import torch
import json

from torch.utils.data import Dataset, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ver = "EleutherAI/gpt-neo-125M"


def generate_input(sentence, paraphrases):
    return ["<s>" + sentence + "</s>>>><p>" + paraphrase + "</p>" for paraphrase in paraphrases]


class ParaphraseDataset(Dataset):
    def __init__(self, dataset_json, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for sample in dataset_json:
            for prompt in generate_input(sample["sentence"], sample["paraphrases"]):
                encodings_dict = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
                self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def train_model(dataset_path):
    """
    Train paraphrase model
    :param dataset_path: path to the train dataset
    :return:
    """
    num_epochs = 5
    tokenizer = AutoTokenizer.from_pretrained(model_ver, pad_token="<|pad|>")
    model = AutoModelWithLMHead.from_pretrained(model_ver).to(device)
    model.resize_token_embeddings(len(tokenizer))

    with open(dataset_path, "r") as f:
        initial_dataset = json.load(f)["data"]
    initial_dataset_token_len = [
        generate_input(sample["sentence"], sample["paraphrases"]) for sample in initial_dataset
    ]
    initial_dataset_token_len = [_ for i in range(len(initial_dataset_token_len)) for _ in initial_dataset_token_len[i]]
    max_length = max([len(tokenizer.encode(sample)) for sample in initial_dataset_token_len])
    dataset = ParaphraseDataset(initial_dataset, tokenizer, max_length=max_length)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    training_args = TrainingArguments(
        output_dir=f"{model_ver}/",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        warmup_steps=100,
        save_steps=5000,
        logging_steps=5000,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda data: {
            "input_ids": torch.stack([datum[0] for datum in data]),
            "attention_mask": torch.stack([datum[1] for datum in data]),
            "labels": torch.stack([datum[0] for datum in data]),
        },
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    trainer.save_model()


def generate_paraphrase(model_path, sentence, num_of_paraphrases):
    """
    Generate [num_of_paraphrases] paraphrases for given sentence
    :param model_path: path to a pretrained model parameters
    :param sentence:
    :param num_of_paraphrases:
    :return:
    """
    model_trained = GPTNeoForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_ver)
    input = tokenizer("<s>" + sentence + "</s>>>><p>", return_tensors="pt").to(device)

    results = []
    for i in range(num_of_paraphrases):
        gen_tokens = model_trained.generate(
            **input,
            do_sample=True,
            temperature=0.7,
            max_length=256,
            top_p=0.95,
        )
        results.append(tokenizer.decode(gen_tokens[0], skip_special_tokens=True).split("<p>")[1].split("</p>")[0])
    return results
