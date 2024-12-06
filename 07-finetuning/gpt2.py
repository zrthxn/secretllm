import os
from pathlib import Path
from upycli import command
from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline


@command
def stories(
        config: str,
        output_directory: str,
        epochs: int = 5,
        batch_size: int = 32,
        cache_dir: str = ".",
        context_length: int = 128,
        resume_from: str = None):
    
    name = "stories-gpt2-large" if config == "gpt2-large" else "stories-gpt2"
    
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    dataset = [ data["text"] for data in dataset["train"] ]
    
    os.makedirs(f"{output_directory}/{name}/tokenizer", exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"{output_directory}/{name}/tokenizer")
    
    train_dataset = tokenizer(dataset, add_special_tokens=True, truncation=True, max_length=context_length)["input_ids"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    model = GPT2LMHeadModel(
        AutoConfig.from_pretrained(
            config,
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            cache_dir=cache_dir
        ))

    args = TrainingArguments(
        output_dir=f"{output_directory}/{name}",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=8,
        logging_steps=5_000,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        resume_from_checkpoint=resume_from,
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset)

    print(f"GPT-2 size: {model.num_parameters()/1000**2:.1f}M parameters")
    trainer.train()
    trainer.save_model(f"{output_directory}/{name}")


@command
def finetune(
        pretrained_model_name_or_path: str,
        dataset_name_or_path: str,
        output_directory: str,
        epochs: int = 5,
        batch_size: int = 32,
        cache_dir: str = "."):
    
    name = "stories-gpt2-large" if config == "gpt2-large" else "stories-gpt2"
    
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    dataset = [ data["text"] for data in dataset["train"] ]
    
    os.makedirs(f"{output_directory}/{name}/tokenizer", exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    
    train_dataset = tokenizer(dataset, add_special_tokens=True, truncation=True, max_length=context_length)["input_ids"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f"{output_directory}/{name}",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=8,
        logging_steps=5_000,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        resume_from_checkpoint=resume_from,
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset)

    print(f"GPT-2 size: {model.num_parameters()/1000**2:.1f}M parameters")
    trainer.train()
    trainer.save_model(f"{output_directory}/{name}")


@command
def prompt(pretrained_model: Path, max_new_tokens: int = 250, device: torch.device = torch.device("cuda")):
    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model / "tokenizer")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    
    while True:
        prompt = input(">>> Prompt: ")
        print(pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"], "\n")
