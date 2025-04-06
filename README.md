# 🎮 Gomoku AI using Custom Convolutional Neural Network

Welcome to **Gomoku AI**, a strategic board game project powered by a custom **Convolutional Neural Network (CNN)** trained on professional gameplay data. This project showcases how deep learning can be applied to outperform traditional AI strategies in classic games like Gomoku.

---

## 🧠 About the Project

This AI agent plays Gomoku using a **custom 2D CNN model** trained on cleaned data from the **2019 Gomoku Tournament (Gomocup)**. The model interprets 15x15 board states and predicts the best possible next move with high accuracy.

Our architecture uses:
- **5 convolutional layers** with 64, 128, 256, 128, 64 filters (7×7 kernel)
- **ReLU** activation in hidden layers
- **Sigmoid** activation in the output layer
- **Binary Cross-Entropy** as the loss function
- **Adam** optimizer for adaptive learning

The final model outputs a **15×15 probability matrix**, and the move with the **highest probability** is selected using `argmax`.

---

## 🧪 Performance

- Trained on **2884 professional matches**
- **Average win rate**: 25 out of 30 games against a Minimax baseline
- Training done using **TensorFlow Keras on Google Colab**
- Saved as a `.h5` model and integrated into live game simulations

---

## 📁 Project Structure

```
Gomoku/
├── policies/                  # Folder containing policy definitions
├── LICENSE.txt                # License file
├── README.md                  # This file
├── __init__.py                # Init module
├── gomoku.py                  # Core Gomoku game logic
├── compete.py                 # Script for simulating player-vs-player matches
├── performance.py             # Benchmarking model vs baseline strategies
```

---

## 🚀 How It Works

### 1. Game Logic (`gomoku.py`)
- Defines the `GomokuState` class for board representation, move validation, game status, scoring, and winning conditions.
- Efficiently tracks board states with NumPy for fast evaluation.

### 2. Model Competition (`compete.py`)
- Simulates matches between AI policies.
- Validates move legality and measures runtime per player.

### 3. Performance Analysis (`performance.py`)
- Runs 30 matches to evaluate AI vs Minimax.
- Generates runtime histograms and performance scores using `matplotlib`.

---

## 📊 Example Usage

Run a simulation:

```bash
python compete.py -b 15 -w 5 -x Submission -o Minimax
```

Evaluate performance over 30 games:

```bash
python performance.py
```

---

## 📌 References

- [GomoCup](http://gomocup.org) — Gomoku AI Tournament
- Harris, C.R., et al. (2020). *Array programming with NumPy*. Nature.
- Abadi, M., et al. (2015). *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems*.

---

## 👥 Contributors

**Ankit Hiremath**  
In collaboration with:  
**Aditya Porwal**, **Titas Ghosh**, **Ameya Jajulwar**, **Saad Shah**

---

## 📄 License

This project is licensed under the terms of the [LICENSE.txt](./LICENSE.txt) file.

---

> 🧩 Future work: Explore **3D CNNs** for richer spatial awareness and integrate **max-pooling** layers for reduced complexity.
