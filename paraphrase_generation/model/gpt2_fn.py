import pandas as pd
from transformers import (AutoModelWithLMHead, Trainer, AutoTokenizer, TrainingArguments, TextDataset,
                          DataCollatorForLanguageModeling)
import json


def extract_questions(paraphrase_dataset_path, save_path):
    with open(paraphrase_dataset_path, 'r') as f:
        paraphrase_dataset = json.load(f)
    result_dataset = [[j['text'] for j in i['sentences']] for i in paraphrase_dataset if len(i['sentences']) > 1]
    with open(save_path, 'w') as f:
        json.dump(result_dataset, f, indent=2)


def train_model(dataset_path, output_dir, model_ver='gpt2', num_epochs=10):
    model = AutoModelWithLMHead.from_pretrained(model_ver)
    tokenizer = AutoTokenizer.from_pretrained(model_ver)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(file_path=dataset_path, tokenzer=tokenizer, block_size=256)
    training_args = TrainingArguments(output_dir=output_dir, num_training_epochs=num_epochs,
                                      per_per_device_train_batch_size=8, warmup_steps=500, save_steps=2000,
                                      logging_steps=10)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset,
                      prediction_loss_only=True)
    trainer.train()
    trainer.save_model()
