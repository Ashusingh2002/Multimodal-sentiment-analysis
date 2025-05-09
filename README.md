# Multimodal-sentiment-analysis

This project demonstrates a basic implementation of a **multimodal sentiment analysis model** using synthetic data. The model takes both **text** and **image** inputs and predicts binary sentiment (positive or negative). It serves as a minimal working prototype for experimenting with **multimodal learning**, combining the power of BERT (for text) and ResNet (for images).

---

## Motivation

Multimodal learning enables models to understand and process multiple forms of input—like text, images, and audio—simultaneously. It mimics human perception more closely and is key for tasks like emotion recognition, fake news detection, and content moderation.

This implementation provides a simplified yet functional pipeline to understand the core architecture of multimodal learning systems using **synthetic data**. It is ideal for experimentation and gaining intuition before scaling up to real-world datasets.

---

## Project Structure

### 1. Synthetic Dataset (`SyntheticDataset` Class)

We simulate a dataset containing:
- Random sentences like _"This is sample text 0"_
- Randomly generated RGB images
- Binary sentiment labels (0 or 1)

The dataset class handles:
- Tokenizing text using BERT
- Preprocessing images using torchvision transforms
- Generating labels

This setup mimics a real-world multimodal dataset in a controlled, testable environment.

---

### 2. Model Architecture (`MultimodalSentimentModel`)

The model is built using:
- **BERT** (`bert-base-uncased`) for extracting text features
- **ResNet-18** for extracting image features (with the final classification layer replaced)
- A fusion mechanism that **concatenates** BERT and ResNet outputs
- A feedforward classifier for binary sentiment prediction

The architecture is modular, making it easy to extend or swap out components.

---

### 3. Training and Evaluation

Two key functions handle training and evaluation:

- `train(model, dataloader, ...)`: Trains the model over epochs using backpropagation
- `evaluate(model, dataloader, ...)`: Measures loss and accuracy on validation/test sets

**Loss function**: Binary Cross Entropy with Logits  
**Optimizer**: AdamW

---

## How to Run

1. Install dependencies:
   ```bash
   pip install torch torchvision transformers
2. Run the main script:
   ```bash
   python main.py
3. Monitor training and evaluation logs in the console.

## Next Steps

To make this more applicable to real-world use cases:

  Replace the synthetic dataset with real multimodal datasets like CMU-MOSEI, MOSI, or SIMS
  Add audio processing modules for a full tri-modal pipeline
  Experiment with attention-based fusion methods or transformer-based architectures for all modalities

## Acknowledgements
  HuggingFace Transformer
  PyTorch
  Torchvision


 
