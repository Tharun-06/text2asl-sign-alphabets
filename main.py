import sys
import time
import config
from data_utils import load_asl_alphabet_dataset
from input_utils import get_youtube_subtitles
from generation_utils import generate_gloss_from_text
from gui_utils import display_gloss_sequentially

if __name__ == "__main__":
    print("=" * 40)
    print(" ASL Gloss Generation & Fingerspelling Viewer (BART - Sequential) ")
    print(" (Microphone Input Disabled) ")
    print(" (Single Run - Exits After Display) ")
    print("=" * 40)
    print(f"Using Device: {config.DEVICE}")
    print(f"Fine-tuned Model: {config.FINETUNED_MODEL_DIR}")
    print(f"ASL Alphabet Dataset: {config.ASL_ALPHABET_DATASET_NAME}")
    print(f"Max Input Length (Tokens): {config.MAX_INPUT_LENGTH}")
    print("-" * 40)

    asl_image_map = load_asl_alphabet_dataset()
    if not asl_image_map:
        print("\nCritical Error: Could not load ASL Alphabet images. Exiting.")
        sys.exit(1)

    print("\nChoose input method:")
    print("  1: YouTube Video URL (Subtitles)")
    print("  2: Enter Text Directly")
    print("  Q: Quit")

    choice = input("Enter your choice (1, 2, or Q): ").strip().upper()

    english_input_text = None
    generated_gloss = None

    if choice == '1':
        video_url = input("Enter YouTube Video URL: ").strip()
        if video_url:
            english_input_text = get_youtube_subtitles(video_url)
        else:
            print("No URL entered.")
    elif choice == '2':
        english_input_text = input("Enter English text directly: ").strip()
        if not english_input_text:
            print("No text entered.")
    elif choice == 'Q':
        print("Exiting program.")
        sys.exit(0)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    if english_input_text:
        generated_gloss = generate_gloss_from_text(english_input_text)
        if not generated_gloss:
            print("Gloss generation failed.")
    else:
        print("Could not obtain input text from the selected source.")

    if generated_gloss:
        print("\n--- Starting Sequential GUI Display ---")
        display_gloss_sequentially(generated_gloss, asl_image_map)
        print("--- GUI Display Function Finished ---")
    elif english_input_text:
        print("Cannot display gloss because generation failed.")
    else:
        print("Cannot display gloss because input failed.")

    print("\n--- Script Finished ---")
