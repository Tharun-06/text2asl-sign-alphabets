# Text-Gloss-ASL-Sign-Alphabets

![Model](https://img.shields.io/badge/Model-BART--base-blue)
![Dataset](https://img.shields.io/badge/Dataset-ASLG--PC12-brightgreen)
![BLEU Score](https://img.shields.io/badge/BLEU~48-success)
![Platform](https://img.shields.io/badge/Colab-T4_GPU-yellow)
![UI](https://img.shields.io/badge/UI-Tkinter%20%7C%20Flask-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📘 Abstract

This project presents a complete pipeline for translating English text into American Sign Language (ASL) sign alphabets. A BART-base transformer was fine-tuned on the ASLG-PC12 dataset for English-to-ASL gloss translation, achieving a validation BLEU score of ~48. The model was trained and optimized on Google Colab using a T4 GPU. To make the system user-friendly, a Python application was developed that accepts both YouTube captions and direct text input. Preprocessing steps include regex-based cleaning to normalize input data. A rule-based parser interprets the generated gloss into fingerspelled ASL components, which are then rendered visually using ASL alphabet images. These are sequentially displayed via a GUI built using Tkinter (for desktop) and Flask (for web). The project addresses real-world challenges such as domain mismatch, output formatting inconsistencies, and graphical rendering of dynamic image sequences.
