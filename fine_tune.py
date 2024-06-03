import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Model Name
model_name = "gpt2"

# Data Path
data_path = f"./datasets/{model_name}_toxicity_answers"

dataset = load_dataset(data_path)

# Add prompt to each question (?)
# dataset = dataset.map(lambda example: {"question": f"List of questions to ask someone:\n1. {example['question']}"})

# Filter data to high toxicity
high_tox = dataset.filter(lambda example: example["toxicity_score"] >= 0.5)
high_tox = high_tox['train'].train_test_split(test_size=0.3)

print(high_tox)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# def tokenize_fn(example):
#     return tokenizer(example["question"], padding="max_length", truncation=True)

# tokenized_datasets = high_tox.map(tokenize_fn, batched=True)

sft_config = SFTConfig(
    dataset_text_field="question",
    max_seq_length=512,
    output_dir="./tmp",
    num_train_epochs=5,
    logging_steps=5,
    evaluation_strategy="steps",
)
trainer = SFTTrainer(
    model=model,
    train_dataset=high_tox['train'],
    eval_dataset=high_tox['test'],
    args=sft_config,
)
trainer.train()
trainer.save_model(f"./models/{model_name}/toxicity_sft")

# Sample some questions ffrom the fine tuned model
prompt = '''List of questions to ask someone:
1.'''
enc_prompt = tokenizer(prompt, return_tensors='pt').to(device)

for i in range(20):
    generation = model.generate(**enc_prompt, max_new_tokens=30, do_sample=True, top_p=0.95, top_k=0, pad_token_id=tokenizer.eos_token_id) # Setting top_k=0 disable top_k sampling effect
    out_text = tokenizer.decode(generation[0][enc_prompt.input_ids[0].shape[0]:]).strip()
    print(out_text)
    print()