
import torch
import math

DATASET_NAME = "aslg_pc12"
SUBSET_SIZE = None
VALIDATION_SPLIT_PERCENTAGE = 10

BASE_MODEL_CHECKPOINT = "facebook/bart-base"
OUTPUT_DIR = "./bart-asl-gloss-translator-colab/"
FINETUNED_MODEL_DIR = OUTPUT_DIR 
LOGGING_DIR = f"{OUTPUT_DIR}/logs"

ASL_ALPHABET_DATASET_NAME = "Marxulia/asl_sign_languages_alphabets_v03"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
GENERATION_NUM_BEAMS = 4
GENERATION_MAX_LENGTH_FACTOR = 1.2

BATCH_SIZE = 16
GRAD_ACCUMULATION_STEPS = 2

LEARNING_RATE = 5e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
FP16_ENABLED = torch.cuda.is_available()
OPTIMIZER = "adamw_torch"

EVAL_BATCH_SIZE = BATCH_SIZE * 2
METRIC_FOR_BEST_MODEL = "bleu"
GREATER_IS_BETTER = True

EVAL_STRATEGY = "steps"
_estimated_train_size = 87710 * (1 - VALIDATION_SPLIT_PERCENTAGE / 100.0) if SUBSET_SIZE is None else SUBSET_SIZE * (1 - VALIDATION_SPLIT_PERCENTAGE / 100.0)
_steps_per_epoch = math.ceil(_estimated_train_size / (BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
EVAL_STEPS = max(100, math.ceil(_steps_per_epoch / 2))

SAVE_STRATEGY = "steps"
SAVE_STEPS = EVAL_STEPS
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 50
PREDICT_WITH_GENERATE = True
LOAD_BEST_MODEL_AT_END = True
REPORT_TO = "tensorboard"
PUSH_TO_HUB = False

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3

IMAGE_DISPLAY_SIZE = (100, 100)
WINDOW_TITLE = "ASL Gloss Fingerspelling Viewer (BART)"
MAX_IMAGES_PER_ROW = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
