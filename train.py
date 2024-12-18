import torch

from dataset import RouterCollectionDataset

import os
from functools import partial

from accelerate import Accelerator

from torch.utils.data import DataLoader
from accelerate import PartialState
# PartialState().is_main_process

def collate_fn(batch, tokenizer, is_hard):
    texts = []
    labels = []
    for item in batch:
        texts.append(item["prompt"])
        labels.append(item["passed"])
    inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")
    labels = torch.tensor(labels, dtype=torch.long if is_hard else torch.float32)
    return inputs, labels


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        scheduler,
        train_data_folder,
        save_path,
        train_batch_size=8,
        train_type="hard",
        eval_data_folder=None,
        eval_batch_size=8,
        eval_type="soft",
    ):
        self.accelerator = Accelerator()
    
        assert train_type in ["hard", "soft"]
        assert eval_type in ["hard", "soft"]

        self.train_type = train_type
        self.eval_type = eval_type

        datasets = {}
        dls = {}

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(model, optimizer, scheduler)

        if train_data_folder:
            train_collate_fn = partial(collate_fn, tokenizer=tokenizer, is_hard=(train_type == "hard"))
            datasets["train"] = RouterCollectionDataset(train_data_folder)
            dls["train"] = DataLoader(datasets["train"], collate_fn=train_collate_fn, batch_size=train_batch_size, shuffle=True)
            self.train_dl = self.accelerator.prepare(dls["train"])
        else:
            self.train_dl = None

        if eval_data_folder:
            eval_collate_fn = partial(collate_fn, tokenizer=tokenizer, is_hard=(eval_type == "hard"))
            datasets["eval"] = RouterCollectionDataset(eval_data_folder)
            dls["eval"] = DataLoader(datasets["eval"], collate_fn=eval_collate_fn, batch_size=eval_batch_size, shuffle=False)
            self.eval_dl = self.accelerator.prepare(dls["eval"])
        else:
            self.eval_dl = None

        self.save_path = save_path
        self.last_eval_losses = []

    def _step(self, batch, loss_type):
        inputs, labels = batch
        logits = self.model(**inputs)
        if loss_type == "hard":
            loss_list = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        else:
            loss_0_list = torch.nn.functional.cross_entropy(logits, torch.zeros(labels.size(), dtype=torch.long), reduction="none")
            loss_1_list = torch.nn.functional.cross_entropy(logits, torch.ones(labels.size(), dtype=torch.long), reduction="none")
            loss_list = labels * loss_1_list + (1 - labels) * loss_0_list
        
        return loss_list.mean()

    def train_iter(self):
        for batch in self.train_dl:
            yield batch

    def save_model(self, special_name):
        if PartialState().is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save(os.path.join(self.save_path, special_name))
            print(f"Model successfully saved to {os.path.join(self.save_path, special_name)}")
    
    def eval(self):
        assert self.eval_dl is not None

        if PartialState().is_main_process:
            print("Start evaluation.")
        
        with torch.no_grad():
            self.model.eval()

            loss_numerator = 0.
            loss_denominator = 0

            for batch in self.eval_dl:
                loss = self._step(batch, self.eval_type).item()

                loss_list = self.accelerator.gather_for_metrics(loss)

                loss_numerator += sum(loss_list)
                loss_denominator += len(loss_list)

                if PartialState().is_main_process:
                    print(f"Estimated eval loss = {loss_numerator / loss_denominator}", flush=True)
            
            if PartialState().is_main_process:
                print("Evaluation finished.")
                print(f"Final eval loss = {loss_numerator / loss_denominator}", flush=True)
            
        return loss_numerator / loss_denominator


    def train(self, micro_num=1, save_every=1000, max_epochs=1):
        self.epoch = 0
        train_iter = self.train_iter()
        step = 0
        while True:
            try:
                self.model.train()
                for step, batch in zip(range(save_every*micro_num), train_iter):
                    loss = self._step(batch, self.train_type)
                    self.accelerator.backward(loss)

                    print(f"epoch {self.epoch} step {step}: loss = {loss.item()} on process {self.accelerator.local_process_index}", flush=True)

                    if step % micro_num == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                                        
                self.save_model(f"epoch_{self.epoch}_step_{step}")

            except StopIteration:

                self.optimizer.zero_grad()

                self.epoch += 1
                self.save_model(f"epoch_{self.epoch}")
                
                if self.epoch >= max_epochs:
                    return
                
                train_iter = self.train_iter()

            self.accelerator.wait_for_everyone()

            if self.eval_dl is not None:
                loss = self.eval()
                self.last_eval_losses.append(loss)
            



