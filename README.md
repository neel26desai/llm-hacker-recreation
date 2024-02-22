# llm-hacker-recreation
This repo is my attempt at recreating and experimenting with LLMs as shown by Jermey Howard in his YouTube video "Hacker's Guide to Language Model"

## File Structure and their Contents
1. fastai_llmhacker.ipynb
 - Using OpenAI API and models on Huggingface and running them locally. The following topics were explored in the notebook
   - Using OpenAI API for chat completion and creating a local version of the code interpreter
   - Using meta-llama/Llama-2-7b-h with 8bit float parameters
   - Using meta-llama/Llama-2-7b-h with torch.bfloat16 parameters
   - Using the quantized version of LLama 2
   - Using Instruction-tuned LLM StableBeluga with 7 billion parameters
   - Using fine-tuned version Llama2 OpenOrca-Platypus2 with 13 billion parameters GPTQ version
   - Running LLama2 on CPU using llama.ccp
     
2. finetuning_with_axoltol.ipyng
 - Fine-tuning llama 2 model on SQL data, i.e generating SQL queries from prompts and context, using axoltol
3. sql.yml
 - the configuration script used to configure the finetuning used in finetuning_with_axoltol.ipyng
4. exploredata.py
 - python script used to explore the training data  used for finetuning llama2 model, used in finetuning_with_axoltol.ipyng
5. testing_the_fine_tuned_model.py
  - python script used for making inference on data using the finetuned llama2 model, used in finetuning_with_axoltol.ipyng

## References
  - https://github.com/openai/openai-python
  - https://github.com/fastai/lm-hackers/blob/main/lm-hackers.ipynb
  - https://github.com/OpenAccess-AI-Collective/axolotl
  - https://www.kaggle.com/code/alaajah/creating-virtual-environment-on-google-colab
