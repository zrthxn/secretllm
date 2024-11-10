from sys import argv
from argparse import ArgumentParser
from datasets import load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig
from transformers.trainer import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling


parser = ArgumentParser()
parser.add_argument("--output_directory", type=str)
parser.add_argument("--language", type=str)
args, = parser.parse_known_args(argv[1:])

DIRECTORY = args.output_directory
LANG = args.language

dataset = load_dataset("wikimedia/wikipedia", f"20231101.{LANG}", cache_dir=f"{DIRECTORY}/data")
dataset = [ data["text"] for data in dataset["train"] ]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
VOCAB_SIZE = 50_000
tokenizer.train_from_iterator(
    dataset,
    vocab_size=VOCAB_SIZE,
    min_frequency=2, 
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

tokenizer.enable_truncation(max_length=512)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")))
tokenizer.save_model(f"{DIRECTORY}/{LANG}-tokenizer")

tokenizer = RobertaTokenizerFast.from_pretrained(f"{DIRECTORY}/{LANG}-tokenizer", max_len=512)

model = RobertaForMaskedLM(
    RobertaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    ))


train_dataset = tokenizer(dataset, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=f"{DIRECTORY}/{LANG}-roberta",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model(f"{DIRECTORY}/{LANG}-roberta")
