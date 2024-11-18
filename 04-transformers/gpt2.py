from upycli import command
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import notebook_login, Repository, get_full_repo_name

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator

from transformers import get_scheduler
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline


@command
def wikipedia(
        language: str,
        config: str,
        output_directory: str,
        epochs: int = 5,
        batch_size: int = 32,
        cache_dir: str = ".",
        context_length: int = 128):
    
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}", cache_dir=cache_dir)
    dataset = [ data["text"] for data in dataset["train"] ]
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"{output_directory}/{language}-gpt2/tokenizer")
    
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
        output_dir=f"{output_directory}/{language}-gpt2",
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
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset)

    print(f"GPT-2 size: {model.num_parameters()/1000**2:.1f}M parameters")
    trainer.train()
    trainer.save_model(f"{output_directory}/{language}-gpt2")


@command
def stories(
        config: str,
        output_directory: str,
        epochs: int = 5,
        batch_size: int = 32,
        cache_dir: str = ".",
        context_length: int = 128):
    
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    dataset = [ data["text"] for data in dataset["train"] ]
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"{output_directory}/stories-gpt2/tokenizer")
    
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
        output_dir=f"{output_directory}/stories-gpt2",
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
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset)

    print(f"GPT-2 size: {model.num_parameters()/1000**2:.1f}M parameters")
    trainer.train()
    trainer.save_model(f"{output_directory}/stories-gpt2")


@command
def prompt(pretrained_model: Path, max_new_tokens: int = 250, device: torch.device = torch.device("cuda")):
    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model / "tokenizer")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    
    while True:
        prompt = input(">>> Prompt: ")
        print(pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"], "\n")
    

    
# ====================================================
# Custom Training ------------------------------------

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


@command
def custom_train():
    tokenized_dataset.set_format("torch")
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(tokenized_dataset["valid"], batch_size=32)

    weight_decay = 0.1

    model = GPT2LMHeadModel(config)
    optimizer = AdamW(get_grouped_params(model), lr=5e-4)

    accelerator = Accelerator(fp16=True)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1_000,
        num_training_steps=num_training_steps,
    )

    model_name = "codeparrot-ds-accelerate"
    repo_name = get_full_repo_name(model_name)
    output_dir = "codeparrot-ds-accelerate"
    repo = Repository(output_dir, clone_from=repo_name)

    evaluate()

    gradient_accumulation_steps = 8
    eval_steps = 5_000

    model.train()

    completed_steps = 0
    for epoch in range(num_train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
            logits = model(batch["input_ids"]).logits
            loss = loss
            
            if step % 100 == 0:
                accelerator.print({
                    "lr": get_lr(),
                    "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                })
            
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate()
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
