import torch
from transformers import pipeline, BloomTokenizerFast, BloomForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sklearn.model_selection import GridSearchCV

data_dir = 'combined_data/combined.txt'
eval_dir = 'validation_data/valid.txt'

tokenizer = AutoTokenizer.from_pretrained('Naseej/noon-7b')
model = AutoModelForCausalLM.from_pretrained('Naseej/noon-7b')

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=data_dir,
    block_size=128
)

valid_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=eval_dir,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./train_output',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    fp16=True,
    fp16_opt_level="02"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_dataset = train_dataset.map(lambda example: {k: v.to(device) for k, v in example.items()})
valid_dataset = valid_dataset.map(lambda example: {k: v.to(device) for k, v in example.items()})


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

hyperparameters = {
    "num_train_epochs": [3, 5, 7],
    "per_device_train_batch_size": [8, 16, 32],
    "learning_rate": [1e-4, 5e-5, 1e-5],
}


grid_search = GridSearchCV(trainer, hyperparameters, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit()
best_hyperparameters = grid_search.best_params_

trainer.train()

generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


text="قولي يسطا فين عربيتي ونبي"
prompt = f'Instruction:\n{text}\n\nResponse:'

response = generation_pipeline(prompt,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,
    num_beams=4,
    max_length=500,
    top_p=0.1,
    top_k=20,
    repetition_penalty=3.0,
    no_repeat_ngram_size=3)[0]['generated_text']

print(response)


