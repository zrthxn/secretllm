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
        config: Path,
        output_directory: Path,
        learning_rate: float = 5e-4,
        epochs: int = 5,
        batch_size: int = 32,
        cache_dir: Path = ".",
        context_length: int = 128,
        resume_from: Path = None):
    
    name = "stories-gpt2-large" if config == "gpt2-large" else "stories-gpt2"
    (output_directory / name / "tokenizer").mkdir(exist_ok=True)
    
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    dataset = [ data["text"] for data in dataset["train"] ]
    
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
        learning_rate=learning_rate,
        save_steps=5_000,
        fp16=True,
        resume_from_checkpoint=resume_from.absolute(),
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
def instruct(
        pretrained_model_name_or_path: Path,
        output_directory: Path,
        learning_rate: float = 5e-5,
        epochs: int = 5,
        batch_size: int = 32,
        context_length: int = 128,
        cache_dir: Path = "."):
        
    output_directory.mkdir(exist_ok=True)

    dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise", cache_dir=cache_dir)
    dataset = [
        f"{data['prompt']}\n\n{data['chosen']}"
        for data in dataset["train"] ]
        
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = tokenizer(dataset, add_special_tokens=True, truncation=True, max_length=context_length)["input_ids"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=8,
        logging_steps=5_000,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        save_steps=5_000,
        fp16=True,
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset)

    print(f"GPT-2 size: {model.num_parameters()/1000**2:.1f}M parameters")
    trainer.train()
    trainer.save_model(output_directory)


@command
def finetune(
        pretrained_model_name_or_path: Path,
        output_directory: Path,
        learning_rate: float = 5e-4,
        epochs: int = 5,
        batch_size: int = 32,
        context_length: int = 128,
        cache_dir: Path = "."):
        
    output_directory.mkdir(exist_ok=True)

    dataset = load_dataset("Hieu-Pham/kaggle_food_recipes", cache_dir=cache_dir)
    dataset = [ data["Instructions"] for data in dataset["train"] ]
        
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    
    # Training setup
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("Trainable percentage: {:.2f}%".format(trainable_percentage))
    
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = tokenizer(dataset, add_special_tokens=True, truncation=True, max_length=context_length)["input_ids"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=8,
        logging_steps=5_000,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        save_steps=5_000,
        fp16=True,
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset)

    print(f"GPT-2 size: {model.num_parameters()/1000**2:.1f}M parameters")
    trainer.train()
    trainer.save_model(output_directory)


@command
def prompt(pretrained_model: Path, max_new_tokens: int = 250, device: torch.device = torch.device("cuda")):
    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model / "tokenizer")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    
    while True:
        prompt = input(">>> Prompt: ")
        print(pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"], "\n")
