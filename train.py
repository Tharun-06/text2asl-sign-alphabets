import numpy as np
import evaluate
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import os
import sys

import config
from data_utils import load_asl_dataset, tokenize_datasets_for_training
from model_utils import load_tokenizer, load_model

try:
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    metrics_loaded = True
except Exception as e:
    print(f"Warning: Could not load evaluation metrics: {e}")
    metrics_loaded = False

try:
    global_tokenizer = load_tokenizer(config.BASE_MODEL_CHECKPOINT)
except Exception as e:
    print(f"Fatal Error: Could not load tokenizer '{config.BASE_MODEL_CHECKPOINT}'. Exiting.")
    print(e)
    sys.exit(1)

def compute_metrics(eval_pred):
    if not metrics_loaded: return {}
    predictions, labels = eval_pred
    tokenizer = global_tokenizer
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels_bleu = [[label.strip()] for label in decoded_labels]
    decoded_labels_rouge = [label.strip() for label in decoded_labels]
    result = {}
    try:
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
        result["bleu"] = bleu_result["score"]
    except Exception as e: result["bleu"] = 0.0; print(f"Warn: BLEU failed: {e}")
    try:
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels_rouge)
        result.update({key: value for key, value in rouge_result.items()})
    except Exception as e: print(f"Warn: ROUGE failed: {e}")
    try:
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
    except Exception as e: result["gen_len"] = 0.0; print(f"Warn: Gen len failed: {e}")
    return {k: round(v, 4) for k, v in result.items()}

def main():
    tokenizer = global_tokenizer
    model = load_model(config.BASE_MODEL_CHECKPOINT, tokenizer)
    model.to(config.DEVICE)

    raw_datasets = load_asl_dataset(subset_size=config.SUBSET_SIZE)
    tokenized_datasets = tokenize_datasets_for_training(raw_datasets, tokenizer)

    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")
    test_dataset = tokenized_datasets.get("test")

    if train_dataset is None:
        print("Fatal Error: 'train' split not found. Cannot train.")
        return

    current_eval_strategy = config.EVAL_STRATEGY
    if eval_dataset is None and current_eval_strategy != "no":
        if train_dataset:
            split_percentage = config.VALIDATION_SPLIT_PERCENTAGE / 100.0
            try:
                split_dataset = tokenized_datasets["train"].train_test_split(test_size=split_percentage, seed=42)
                tokenized_datasets["train"] = split_dataset["train"]
                tokenized_datasets["validation"] = split_dataset["test"]
                eval_dataset = tokenized_datasets["validation"]
            except Exception as e:
                 print(f"Error splitting dataset: {e}. Disabling evaluation.")
                 current_eval_strategy = "no"
                 eval_dataset = None
        else:
            print("Error: 'train' split missing. Disabling evaluation.")
            current_eval_strategy = "no"
            eval_dataset = None

    if current_eval_strategy != "no" and eval_dataset is None:
         print("Error: Still no evaluation dataset available. Cannot evaluate.")
         return

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        eval_strategy=current_eval_strategy,
        eval_steps=config.EVAL_STEPS if current_eval_strategy == "steps" else None,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        save_strategy=config.SAVE_STRATEGY,
        save_steps=config.SAVE_STEPS if config.SAVE_STRATEGY == "steps" else None,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        num_train_epochs=config.EPOCHS,
        predict_with_generate=config.PREDICT_WITH_GENERATE,
        fp16=config.FP16_ENABLED,
        logging_dir=config.LOGGING_DIR,
        logging_steps=config.LOGGING_STEPS,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END and current_eval_strategy != "no",
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL if current_eval_strategy != "no" else None,
        greater_is_better=config.GREATER_IS_BETTER,
        optim=config.OPTIMIZER,
        report_to=config.REPORT_TO,
        push_to_hub=config.PUSH_TO_HUB,
    )

    callbacks = []
    if config.USE_EARLY_STOPPING and current_eval_strategy != "no":
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )
        callbacks.append(early_stopping)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=eval_dataset if current_eval_strategy != "no" else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if current_eval_strategy != "no" else None,
        callbacks=callbacks,
    )

    try:
        resume_from_checkpoint = config.OUTPUT_DIR if os.path.isdir(config.OUTPUT_DIR) and any(fname.startswith("checkpoint-") for fname in os.listdir(config.OUTPUT_DIR)) else None
        if resume_from_checkpoint:
             print(f"Resuming training from latest checkpoint in {resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        raise

    if test_dataset is not None:
        try:
            test_metrics = trainer.evaluate(
                eval_dataset=test_dataset, metric_key_prefix="test",
                max_length=int(config.MAX_TARGET_LENGTH * config.GENERATION_MAX_LENGTH_FACTOR),
                num_beams=config.GENERATION_NUM_BEAMS
            )
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            print(test_metrics)
        except Exception as e:
            print(f"Could not evaluate on test set: {e}")
    else:
        print("\nNo 'test' split found for final evaluation.")

    print("\n--- Training Script Finished ---")

if __name__ == "__main__":
    import config
    main()
