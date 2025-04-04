from datasets import load_dataset, DatasetDict, Dataset
from PIL import Image
from transformers import PreTrainedTokenizer
import config

def load_asl_alphabet_dataset(dataset_name: str = config.ASL_ALPHABET_DATASET_NAME) -> dict[str, Image.Image]:
    print(f"--- Loading ASL Alphabet dataset: {dataset_name} ---")
    letter_to_image_map = {}
    try:
        dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        print("Dataset loaded successfully.")
        if 'image' not in dataset.features or 'label' not in dataset.features:
            print(f"ERROR: Dataset '{dataset_name}' lacks 'image'/'label' columns.")
            return {}
        print("Mapping letters to images...")
        label_feature = dataset.features['label']
        is_class_label = hasattr(label_feature, 'names')
        int_to_letter = {}
        if is_class_label:
            int_to_letter = {i: name.upper() for i, name in enumerate(label_feature.names)}
        for item in dataset:
            image = item['image']; label_data = item['label']; letter = None
            if is_class_label: letter = int_to_letter.get(label_data)
            elif isinstance(label_data, str): letter = label_data.upper()
            if letter and 'A' <= letter <= 'Z' and isinstance(image, Image.Image):
                if letter not in letter_to_image_map: letter_to_image_map[letter] = image
        print(f"Finished mapping. Found images for {len(letter_to_image_map)} letters.")
        if len(letter_to_image_map) < 26: print("Warning: Did not find images for all 26 letters A-Z.")
        return letter_to_image_map
    except Exception as e:
        print(f"ERROR: Failed to load or process dataset '{dataset_name}': {e}")
        return {}

def load_asl_dataset(dataset_name: str = config.DATASET_NAME, subset_size: int | None = config.SUBSET_SIZE) -> DatasetDict:
    print(f"--- Loading Dataset: {dataset_name} ---")
    try:
        raw_datasets = load_dataset(dataset_name)
        print("Dataset loaded successfully.")
        if subset_size:
            print(f"Using a subset of size {subset_size} for train/validation/test.")
            for split in raw_datasets:
                actual_size = min(subset_size, len(raw_datasets[split]))
                raw_datasets[split] = raw_datasets[split].select(range(actual_size))
            print(f"Subset selection complete: {raw_datasets}")
        return raw_datasets
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        raise

def preprocess_data_for_training(examples: dict, tokenizer: PreTrainedTokenizer, max_input_len: int, max_target_len: int) -> dict:
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_len,
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["gloss"],
            max_length=max_target_len,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_datasets_for_training(raw_datasets: DatasetDict, tokenizer: PreTrainedTokenizer) -> DatasetDict:
     print("\n--- Preprocessing & Tokenizing Dataset for Training (BART) ---")
     preprocess_fn = lambda examples: preprocess_data_for_training(
         examples,
         tokenizer,
         config.MAX_INPUT_LENGTH,
         config.MAX_TARGET_LENGTH
     )
     columns_to_remove = list(raw_datasets["train"].column_names) if "train" in raw_datasets else None
     tokenized_datasets = raw_datasets.map(preprocess_fn, batched=True, remove_columns=columns_to_remove)
     print("Preprocessing complete.")
     if "train" in tokenized_datasets and len(tokenized_datasets["train"]) > 0:
         print("Example of tokenized training data structure:")
         print(tokenized_datasets["train"][0])
     return tokenized_datasets
