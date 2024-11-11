from pathlib import Path
from transformers import RobertaForCausalLM, RobertaTokenizerFast, pipeline


PWD = Path(".").absolute()
LANG = "be"

tokenizer = RobertaTokenizerFast.from_pretrained(f"{PWD}/.checkpoints/{LANG}-tokenizer")
model = RobertaForCausalLM.from_pretrained(f"{PWD}/.checkpoints/{LANG}-roberta")

sentence = "Упершыню выяўлены падчас навуковай "

generate = pipeline("text-generation", model=model, tokenizer=tokenizer, device="mps")
output = generate(sentence, max_new_tokens = 100)

print(output)
