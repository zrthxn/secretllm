from upycli import command
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import notebook_login, Repository, get_full_repo_name

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator

from transformers import get_scheduler
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline


@command
def train(
    context_length: int = 128,
    device: str = torch.device("cpu")
):
    
    split = "train"
    dataset = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
    
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

    raw_datasets = DatasetDict({
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    })

    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True)
        
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        
        return { "input_ids": input_batch }


    tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config)
    print(f"GPT-2 size: {model.num_parameters()/1000**2:.1f}M parameters")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()

@command
def prompt():
    pipe = pipeline(
        "text-generation", model="huggingface-course/codeparrot-ds", device=device
    )

    txt = """\
    # create some data
    x = np.random.randn(100)
    y = np.random.randn(100)

    # create scatter plot with x, y
    """
    print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

    txt = """\
    # create some data
    x = np.random.randn(100)
    y = np.random.randn(100)

    # create dataframe from x and y
    """
    print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

    txt = """\
    # dataframe with profession, income and name
    df = pd.DataFrame({'profession': x, 'income':y, 'name': z})

    # calculate the mean income per profession
    """
    print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

    txt = """
    # import random forest regressor from scikit-learn
    from sklearn.ensemble import RandomForestRegressor

    # fit random forest model with 300 estimators on X, y:
    """
    print(pipe(txt, num_return_sequences=1)[0]["generated_text"])


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
