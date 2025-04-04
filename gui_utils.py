import tkinter as tk
from tkinter import ttk, messagebox, font as tkFont
from PIL import Image, ImageTk
import string
import config

root = None
word_display_label = None
image_display_label = None
current_img_tk = None
after_id = None

def display_gloss_sequentially(gloss_string: str, image_map: dict[str, Image.Image]):
    global root, word_display_label, image_display_label, current_img_tk, after_id
    if not image_map:
        print("ERROR (GUI): ASL Alphabet Image map is empty.")
        messagebox.showerror("Error", "ASL Alphabet Image map is empty.")
        return
    if not gloss_string or gloss_string.isspace():
        print("ERROR (GUI): Empty gloss string provided.")
        messagebox.showerror("Error", "Empty gloss string generated.")
        return
    if root and root.winfo_exists():
        try:
            root.destroy()
        except tk.TclError:
            pass
        root = None
    root = tk.Tk()
    root.title(config.WINDOW_TITLE)
    root.geometry("450x450")
    word_font = tkFont.Font(family="Arial", size=20, weight="bold")
    info_font = tkFont.Font(family="Arial", size=9)
    word_display_label = tk.Label(root, text="Starting...", font=word_font, pady=15, fg="blue")
    word_display_label.pack()
    placeholder_pil = Image.new('RGB', config.IMAGE_DISPLAY_SIZE, color='white')
    placeholder_img_tk = ImageTk.PhotoImage(placeholder_pil)
    image_display_label = tk.Label(root, image=placeholder_img_tk, width=config.IMAGE_DISPLAY_SIZE[0], height=config.IMAGE_DISPLAY_SIZE[1], borderwidth=1, relief="solid")
    image_display_label.pack(pady=5)
    image_display_label.image = placeholder_img_tk
    current_img_tk = placeholder_img_tk
    info_text = f"Displaying Gloss Sequentially\nDataset: {config.ASL_ALPHABET_DATASET_NAME}"
    info_label = tk.Label(root, text=info_text, font=info_font, justify=tk.CENTER, fg="grey")
    info_label.pack(pady=(10, 5))
    gloss_words = gloss_string.split()
    if not gloss_words:
        print("ERROR (GUI): Gloss string contains no words after splitting.")
        messagebox.showerror("Error", "Gloss string contains no words.")
        root.destroy()
        return
    if after_id:
        try:
            root.after_cancel(after_id)
        except tk.TclError:
            pass
        after_id = None
    after_id = root.after(500, process_next_gloss_word, gloss_words, 0, image_map)
    root.mainloop()

def process_next_gloss_word(word_list, word_index, image_map):
    global root, word_display_label, image_display_label, current_img_tk, after_id
    if not root or not root.winfo_exists():
        return
    if word_index < len(word_list):
        current_gloss_word = word_list[word_index]
        word_display_label.config(text=current_gloss_word)
        root.update_idletasks()
        items_to_display = []
        is_complex_or_tag = False
        word_upper = current_gloss_word.upper()
        if '-' in current_gloss_word:
            parts = current_gloss_word.split('-')
            for part in parts:
                part_upper = part.upper()
                if part.isalpha():
                    items_to_display.extend(list(part_upper))
                elif part:
                    items_to_display.append(f"[{part}]")
                    is_complex_or_tag = True
            if not items_to_display:
                is_complex_or_tag = True
                items_to_display.append(f"[{current_gloss_word}]")
        elif len(current_gloss_word) == 1 and 'A' <= word_upper <= 'Z':
            items_to_display = [word_upper]
        elif current_gloss_word.isalpha():
            items_to_display = list(word_upper)
        else:
            is_complex_or_tag = True
            items_to_display.append(f"[{current_gloss_word}]")
        if items_to_display:
            after_id = root.after(config.LETTER_DELAY_MS, display_next_item, items_to_display, 0, word_list, word_index, image_map)
        else:
            after_id = root.after(config.WORD_DELAY_MS, process_next_gloss_word, word_list, word_index + 1, image_map)
    else:
        word_display_label.config(text="- End -")
        placeholder_pil = Image.new('RGB', config.IMAGE_DISPLAY_SIZE, color='white')
        placeholder_img_tk = ImageTk.PhotoImage(placeholder_pil)
        image_display_label.config(image=placeholder_img_tk, text="")
        image_display_label.image = placeholder_img_tk
        current_img_tk = placeholder_img_tk
        after_id = None

def display_next_item(items_in_word, item_index, word_list, word_index, image_map):
    global root, image_display_label, current_img_tk, after_id
    if not root or not root.winfo_exists():
        return
    if item_index < len(items_in_word):
        item = items_in_word[item_index]
        display_image_tk = None
        display_text = ""
        if len(item) == 1 and 'A' <= item <= 'Z':
            letter = item
            pil_image = image_map.get(letter)
            if pil_image:
                try:
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    img_resized = pil_image.resize(config.IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                    display_image_tk = ImageTk.PhotoImage(img_resized)
                    display_text = f"Letter: {letter}"
                except Exception as e:
                    print(f"ERROR (GUI): Error processing image for letter '{letter}': {e}")
                    display_text = f"Error for '{letter}'"
                    display_image_tk = ImageTk.PhotoImage(Image.new('RGB', config.IMAGE_DISPLAY_SIZE, color='red'))
            else:
                display_text = f"No sign for '{letter}'"
                display_image_tk = ImageTk.PhotoImage(Image.new('RGB', config.IMAGE_DISPLAY_SIZE, color='lightgrey'))
        else:
            display_text = f"Part: {item}"
            display_image_tk = ImageTk.PhotoImage(Image.new('RGB', config.IMAGE_DISPLAY_SIZE, color='grey'))
        image_display_label.config(image=display_image_tk, text=display_text, compound='center', font=("Arial", 9))
        image_display_label.image = display_image_tk
        current_img_tk = display_image_tk
        after_id = root.after(config.LETTER_DELAY_MS, display_next_item, items_in_word, item_index + 1, word_list, word_index, image_map)
    else:
        after_id = root.after(config.WORD_DELAY_MS, process_next_gloss_word, word_list, word_index + 1, image_map)
