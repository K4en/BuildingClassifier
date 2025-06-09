# README.md

## 🏗️ AI Building Classifier

This project is a from-scratch convolutional neural network built using PyTorch to classify architectural images. The model begins with simple binary classification (e.g., **residential** vs **industrial**) and is evolving toward a full **multi-class architecture recognition system**, including both **cultural origins** (e.g., Egyptian, Chinese, Greek) and **stylistic classifications** (e.g., Renaissance, Baroque).

What makes this project different is that it mirrors **how humans learn** — through *meaningful memory*. The model saves the most significant examples from each training run (best and worst) and reuses them during future training to simulate a kind of **learning memory**. This concept is known in machine learning as *replay buffering* or *exemplar memory*, but it was developed here from a natural human insight.

---

## 🚀 Features Implemented

- [x] CNN model built from scratch in PyTorch
- [x] DuckDuckGo image scraping with multi-query support
- [x] Training and validation dataset pipeline
- [x] Memory replay buffer (save most & least accurate samples)
- [x] Resume training with memory-injected batches
- [x] Binary classifier (residential vs industrial)

---

## 🧠 Conceptual Design

Rather than just train on more and more data, this model is designed to **remember what mattered**. Each epoch, it saves the 10 most confidently correct and 10 most confusing samples. These are injected into future training runs, ensuring that the model is learning **not just more, but better**.

This concept is based on how humans often return to defining moments in our learning journeys — both successes and failures — to anchor deeper understanding.

---

## 🛣️ Roadmap

### ✅ Phase 1 — Binary Classification
- Residential vs Industrial buildings using scraped data

### 🔜 Phase 2 — Multi-Class Cultural Classification
- Categories like: Egyptian, Chinese, Greek, Islamic, Roman, etc.

### 🔮 Phase 3 — Fine-Grained Architectural Styles
- Renaissance, Baroque, Gothic, Neoclassical, Brutalist, and more

---

## 📂 Project Structure

```
AI_Building_Classifier/
├── Data/               
    ├── Raw/                   # Raw scraped images
    ├── Training/              # Structured train/val folders
    ├── Val/                   # Structured train/val folders
├── models/                    # Saved model checkpoints
├── replay_buffer/             # Stored key images from past runs
├── model.py                   # CNN architecture
├── train.py                   # Basic training loop
├── replay_train.py            # Training with replay buffer
├── evaluate.py                # Model evaluation script
├── download_images.py         # Multi-query DuckDuckGo scraper
├── dataset_splitter.py        # Script to split raw images into train/val
└── JOURNAL.md                 # Thought log and original ideas
```

---
📉 Training Loss Over Time
![training_loss](https://github.com/user-attachments/assets/28a52d68-dd0b-4119-8974-e7915d6c1dc8)
---

## 🧾 License & Attribution

All ideas, structure, and conceptual learning logic designed by [Tamas Kiss](https://github.com/K4en).

Some code co-developed with the help of ChatGPT-4.

Use for learning and experimentation — contributions and forks welcome!
