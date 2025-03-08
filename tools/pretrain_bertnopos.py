import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

class BertNoPositionalEmbedding(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # Remove positional embeddings
        self.bert.embeddings.position_embeddings = None
        
    def forward(self, *args, **kwargs):
        # Ensure position_ids are not used
        kwargs.pop("position_ids", None)
        return super().forward(*args, **kwargs)

# Load dataset (FineWeb)
dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

dataset = dataset.train_test_split(test_size=0.1)

# Load model without positional embeddings
model = BertNoPositionalEmbedding.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    max_steps=100000,
    logging_dir="./logs",
    logging_steps=500,
    report_to="none",  # Disable reporting
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

# Save model and tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
