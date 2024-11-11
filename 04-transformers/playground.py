from pathlib import Path
from transformers import RobertaForCausalLM, RobertaTokenizerFast, RobertaConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import pipeline


PWD = Path(".").absolute()
LANG = "hi"

tokenizer = RobertaTokenizerFast.from_pretrained(f"{PWD}/.checkpoints/{LANG}-tokenizer")
model = RobertaForCausalLM.from_pretrained(f"{PWD}/.checkpoints/{LANG}-roberta",
    config=RobertaConfig.from_pretrained(f"{PWD}/.checkpoints/{LANG}-roberta"))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


sentence = "आराधना करने वाले को भक्त "
batch = data_collator(tokenizer(sentence, return_tensors="pt").input_ids)
print(batch["input_ids"])
print(batch["labels"])

# generate = pipeline("text-generation", model=model, tokenizer=tokenizer, device="mps")
# output = generate(sentence, max_new_tokens = 500)

# print(output)
