from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from peft import PeftModel
ax_model = '/content/drive/MyDrive/qlora/qlora-out'
tokr = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',
                                             torch_dtype=torch.bfloat16, device_map=0)
model = PeftModel.from_pretrained(model, ax_model)
model = model.merge_and_unload()
model.save_pretrained('sql-model')

fmt = """SYSTEM: Use the following contextual information to concisely answer the question.

USER: {}
===
{}
ASSISTANT:"""

def sql_prompt(d): 
  return fmt.format(d["context"], d["question"])

#gett the data 
import datasets
ds =  datasets.load_dataset('knowrohit07/know_sql')

tst = ds['validation'][3]

tst['question'] = 'Get the count of competition hosts by theme.'

print(sql_prompt(tst))

toks = tokr(sql_prompt(tst), return_tensors="pt")
res = model.generate(**toks.to("cuda"), max_new_tokens=250).to('cpu')
print(tokr.batch_decode(res)[0])