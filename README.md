# 🧠 Hangman Transformer AI – Man vs The Machine

An AI that plays Hangman by predicting the next letter using a custom-trained Transformer model and real-time gameplay in a retro-style Streamlit app.

Try it live 👉 [Play Here!](https://hangman-vs-ai.streamlit.app)

---

## 🎯 Project Overview

This project started as a take-home exercise for an interview process and evolved into a full-featured, character-level AI game. The AI plays classic Hangman using a Transformer model trained on masked word samples and gameplay context with no handcrafted logic or heuristics, just pure model-driven inference.

---

## 📈 Performance

- **Training set**: ~84,000 words (SCOWL 70)
- **Test set**: ~8,000 words (SCOWL 60)
- **Gameplay**: Based on a filtered list of ~3,000 hand-cleaned words from the test set
- **Win rate**: Achieves **58% accuracy** on the full test set (completely unseen during training)

---

## 🧠 How It Works

### Transformer Model
A PyTorch Transformer encoder trained to predict the next letter in masked Hangman-style words.

### Features per sample (~3 million generated samples)
- **19 auxiliary features** (e.g., mask structure, guessed letters, remaining guesses)
- **4 multi-hot feature vectors** (vowels, consonants, guesses, misses)
- **N-gram frequency vectors**
- **Masked entropy score** for uncertainty awareness

### Training Pipeline
- PyTorch-based training loop
- Early stopping, validation tracking
- Regularisation: **token dropout**, **label smoothing**, **weight decay**
- Trained on GPU

---

## 🕹️ Gameplay App

Built with [Streamlit](https://streamlit.io), the web app provides:
- Real-time interactive play
- Local model inference (no backend required)
- Retro UI, responsive layout (best played on desktop, mobile works in landscape)
