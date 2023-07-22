from transformers import RobertaTokenizer, TFRobertaForMaskedLM, TFTrainer, TFTrainingArguments
import tensorflow as tf
import optuna

train_dataset = '../combined_data/combined.txt'
eval_dataset = '../validation_data/valid.txt'

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaForMaskedLM.from_pretrained('roberta-base')

training_args = TFTrainingArguments(
    output_dir="./roberta_finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    save_steps=5000,
    logging_dir="./logs",
    logging_steps=100,
    overwrite_output_dir=True,
)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

def model_evaluation(model, eval_dataset):
    # Custom evaluation function to be used during hyperparameter tuning
    # Replace this with your actual evaluation code
    # For language modeling, you can compute perplexity or any other relevant metric.
    results = trainer.evaluate(eval_dataset=eval_dataset)
    return results["eval_loss"]

def objective(trial):
    # Define the search space for hyperparameters
    training_args = TFTrainingArguments(
        output_dir="./roberta_finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", 500, 2000, step=500),
        save_steps=trial.suggest_int("save_steps", 5000, 10000, step=1000),
        logging_dir="./logs",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    eval_loss = model_evaluation(model, eval_dataset)

    return eval_loss



from transformers import RobertaTokenizer, TFRobertaForMaskedLM, TFTrainer, TFTrainingArguments
import tensorflow as tf
import optuna

def model_evaluation(model, eval_dataset):
    # Custom evaluation function to be used during hyperparameter tuning
    # Replace this with your actual evaluation code
    # For language modeling, you can compute perplexity or any other relevant metric.
    results = trainer.evaluate(eval_dataset=eval_dataset)
    return results["eval_loss"]

def objective(trial):
    # Define the search space for hyperparameters
    training_args = TFTrainingArguments(
        output_dir="./roberta_finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", 500, 2000, step=500),
        save_steps=trial.suggest_int("save_steps", 5000, 10000, step=1000),
        logging_dir="./logs",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    # Load the model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = TFRobertaForMaskedLM.from_pretrained('roberta-base')

    # Custom training loop using TFTrainer
    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    trainer.train()

    # Evaluation
    eval_loss = model_evaluation(model, eval_dataset)

    return eval_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    training_args = TFTrainingArguments(
        output_dir="./roberta_finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        warmup_steps=best_params["warmup_steps"],
        save_steps=best_params["save_steps"],
        logging_dir="./logs",
        logging_steps=100,
        overwrite_output_dir=True,
    )

    model = TFRobertaForMaskedLM.from_pretrained('roberta-base')
    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    results = trainer.evaluate(eval_dataset)
    print("Final Evaluation results:", results)

    model.save_pretrained("./roberta_finetuned_model")
    tokenizer.save_pretrained("./roberta_finetuned_model")