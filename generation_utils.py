import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config

def generate_gloss_from_text(
    english_text: str,
    model_path: str = config.FINETUNED_MODEL_DIR,
    tokenizer_path: str = config.FINETUNED_MODEL_DIR,
    device = config.DEVICE,
    max_input_len: int = config.MAX_INPUT_LENGTH,
    max_target_len: int = config.MAX_TARGET_LENGTH,
    num_beams: int = config.GENERATION_NUM_BEAMS,
    max_length_factor: float = config.GENERATION_MAX_LENGTH_FACTOR,
    **generation_kwargs
) -> str | None:
    if not english_text or english_text.isspace():
        print("Error: No input text provided for gloss generation.")
        return None

    print(f"\n--- Generating Gloss for Input Text (BART) ---")
    print(f"Input Text: '{english_text[:max_input_len]}'...")

    model = None
    tokenizer = None
    try:
        print(f"Loading generation model and tokenizer from: {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("Generation model and tokenizer loaded.")
    except OSError as e:
        print(f"Error loading fine-tuned model/tokenizer from {model_path}: {e}")
        print("Ensure the BART model was trained and saved correctly to this path.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return None

    try:
        inputs = tokenizer(
            english_text,
            return_tensors="pt",
            max_length=max_input_len,
            truncation=True,
            padding=True
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        print(f"DEBUG: Tokenized input shape (Batch, SeqLen): {input_ids.shape}")
        if input_ids.shape[1] == max_input_len and len(tokenizer.encode(english_text)) > max_input_len:
             print(f"DEBUG: Input was likely truncated by tokenizer to {input_ids.shape[1]} tokens.")

    except Exception as e:
        print(f"Error during text tokenization: {e}")
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None

    gen_max_length = int(max_target_len * max_length_factor)
    print(f"Generating gloss with num_beams={num_beams}, max_length={gen_max_length}...")
    generated_gloss = None
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=gen_max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        generated_ids = outputs[0]
        generated_gloss = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"--- Generated ASL Gloss ---")
        print(f"'{generated_gloss}'")
        print("-" * 25)

    except Exception as e:
        print(f"Error during gloss generation: {e}")
        generated_gloss = None
    finally:
        if model is not None: del model
        if tokenizer is not None: del tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("Cleaned up generation model from memory.")

    return generated_gloss
