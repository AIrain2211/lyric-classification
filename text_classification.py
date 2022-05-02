# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 18:26
# @Author  : AIrain2211
# @FileName: text_classfication.py

from datasets import load_dataset, load_metric

from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import pipeline

from tqdm.auto import tqdm

class TEXT_CLASSIFICATION:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
        self.model = AutoModelForSequenceClassification.from_pretrained("./models/checkpoint-16000")
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        self.label_dic = {"LABEL_0": "林夕","LABEL_1": "方文山","LABEL_2": "黄霑","LABEL_3": "罗大佑","LABEL_4": "李宗盛","LABEL_5": "黄伟文","LABEL_6": "儿歌"}

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
        model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3", num_labels=6)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_dataset = load_dataset('csv', data_files='data/train_data.csv')['train']
        eval_dataset = load_dataset('csv', data_files='data/eval_data.csv')['train']

        tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=96,
            per_device_eval_batch_size=96,
            num_train_epochs=5,
            weight_decay=0.01,
            save_steps=2000
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

    def evaluate(self):
        from transformers.pipelines.base import KeyDataset

        test_dataset = load_dataset('csv', data_files='data/test_data.csv')['train']
        references = [label['label'] for label in test_dataset]

        predictions = []
        kd = KeyDataset(test_dataset, "text")
        for out in tqdm(self.pipe(kd)):
            predictions.append(int(out['label'][-1]))

        accuracy_metric = load_metric("accuracy")

        results = accuracy_metric.compute(references=references, predictions=predictions)
        print(results)
        return results

if __name__ == '__main__':
    tc = TEXT_CLASSIFICATION()
    tc.train()
    # tc.evaluate()

