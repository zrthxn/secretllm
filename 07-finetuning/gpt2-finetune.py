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
    #Instruction Fine-Tuning
    #dataset = [
    #    f"{data['prompt']}\n\n{data['chosen']}"
    #    for data in dataset["train"] ]
    
    #Role-Based Fine-Tuning
    dataset = [
        f"USER: {data['prompt']}\nSYSTEM: {data['chosen']}"
        for data in dataset["train"]]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path / "tokenizer")
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
    dataset = [ str(data["Instructions"]) for data in dataset["train"] ]
    print(dataset[0])
        
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path / "tokenizer")
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    
    # # Training setup
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()
    
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # trainable_percentage = (trainable_params / total_params) * 100
    
    # print("Total parameters:", total_params)
    # print("Trainable parameters:", trainable_params)
    # print("Trainable percentage: {:.2f}%".format(trainable_percentage))
    
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
