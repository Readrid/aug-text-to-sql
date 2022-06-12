import torch


from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_cosine_schedule_with_warmup

from tqdm import tqdm

from data_processing import SQLDataset
from model.regsqlnet import RegSQLNet


class Text2SQLTrainer(object):
    def __init__(
        self,
        model: RegSQLNet,
        train_dataset: SQLDataset,
        eval_dataset: Dataset = None,
        batch_size: int = 16,
        epochs: int = 5,
        warmup_rate: float = 0.1,
        lr: float = 3e-5,
        decay: float = 0.01,
        verbose: bool = True,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        train_steps_num = int((len(train_dataset) * self.epochs) / self.batch_size)
        warmup_steps = int(len(train_dataset) * warmup_rate)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps_num
        )

        self.optimizer.zero_grad()

    def train(self):
        if self.verbose:
            print("START TRAINING")
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        train_loop = tqdm(train_dataloader) if self.verbose else train_dataloader
        for epoch in range(self.epochs):
            cur_loss = None
            step = 0
            for batch in train_loop:
                self.__to_device(batch)

                self.model.train()
                loss = self.model(**batch)["loss"]
                loss = torch.mean(loss)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                step += 1
                cur_loss = loss.cpu().detach().numpy()
                if step % 100 == 0 and self.verbose:
                    print(f"Epoch: {epoch}, loss: {cur_loss}")

            print(f"Epoch: {epoch}, loss: {cur_loss}")
            if self.eval_dataset is not None:
                self.eval()

        if self.verbose:
            print("TRAIN SUCCESS")

    def eval(self, test_dataset=None):
        if test_dataset is None:
            test_dataset = self.eval_dataset
        # TODO
        pass

    def __to_device(self, data):
        data["input_ids"].to(self.device)
        data["attention_mask"].to(self.device)
        data["token_type_ids"].to(self.device)