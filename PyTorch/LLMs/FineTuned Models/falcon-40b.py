from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import optuna

train_dataset = '../combined_data/combined.txt'
eval_dataset = '../validation_data/valid.txt'

model_id = "tiiuae/falcon-40b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto",
)

training_args = TrainingArguments(
    output_dir="./falcon_40b_finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # You might need to adjust the batch size based on your resources.
    per_device_eval_batch_size=2,
    warmup_steps=500,
    save_steps=5000,
    logging_dir="./logs",
    logging_steps=100,
    overwrite_output_dir=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

#hyperparameter tuning
def model_evaluation(model, tokenizer, eval_dataset):
    # Custom evaluation function to be used during hyperparameter tuning
    # Replace this with your actual evaluation code
    # For text generation, you can compute metrics like perplexity, BLEU, etc.
    results = trainer.evaluate(eval_dataset=eval_dataset)
    return results["eval_loss"]

def objective(trial):
    # Define the search space for hyperparameters
    training_args = TrainingArguments(
        output_dir="./falcon_40b_finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [2, 4, 8]),
        warmup_steps=trial.suggest_int("warmup_steps", 500, 2000, step=500),
        save_steps=trial.suggest_int("save_steps", 5000, 10000, step=1000),
        logging_dir="./logs",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    # Load the model and tokenizer
    model_id = "tiiuae/falcon-40b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto",
    )

    # Create Trainer with the hyperparameters from Optuna
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    trainer.train()

    # Evaluation
    eval_loss = model_evaluation(model, tokenizer, eval_dataset)

    return eval_loss


if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params

    training_args = TrainingArguments(
        output_dir="./falcon_40b_finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=best_params["batch_size"],
        warmup_steps=best_params["warmup_steps"],
        save_steps=best_params["save_steps"],
        logging_dir="./logs",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b-instruct")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    results = trainer.evaluate()
    print("Evaluation results:", results)


    model.save_pretrained("./falcon_40b_finetuned_model")
    tokenizer.save_pretrained("./falcon_40b_finetuned_model")
