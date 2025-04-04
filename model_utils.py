from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizer, PreTrainedModel
import config

def load_tokenizer(model_checkpoint: str = config.BASE_MODEL_CHECKPOINT) -> PreTrainedTokenizer:
    print(f"\n--- Loading Tokenizer for {model_checkpoint} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        print("Warning: Tokenizer missing pad token. Adding default '[PAD]' or eos.")
        if tokenizer.eos_token: tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Tokenizer loaded successfully.")
    return tokenizer

def load_model(model_checkpoint: str = config.BASE_MODEL_CHECKPOINT, tokenizer: PreTrainedTokenizer | None = None) -> PreTrainedModel:
    print(f"\n--- Loading Model: {model_checkpoint} ---")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    if tokenizer:
        if model.config.vocab_size < len(tokenizer):
            print(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        if model.config.pad_token_id != tokenizer.pad_token_id:
            print(f"Setting model's pad_token_id to {tokenizer.pad_token_id}")
            model.config.pad_token_id = tokenizer.pad_token_id
        if "bart" in model_checkpoint.lower() and model.config.decoder_start_token_id is None and tokenizer.eos_token_id is not None:
            print(f"Setting BART model's decoder_start_token_id to EOS ({tokenizer.eos_token_id})")
            model.config.decoder_start_token_id = tokenizer.eos_token_id

    print(f"Model '{model_checkpoint}' loaded successfully.")
    return model
