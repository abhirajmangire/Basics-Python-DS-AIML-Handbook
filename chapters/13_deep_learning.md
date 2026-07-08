# Chapter 13 — Deep Learning Foundations

[← Previous: Model Evaluation](12_model_evaluation.md) | [Back to Index](../README.md) | [Next: NLP →](14_nlp.md)

---

## 13.1 What is Deep Learning?

**Deep Learning** is a subset of Machine Learning using **artificial neural networks** with multiple layers ("deep") to automatically learn hierarchical representations of data — removing much of the need for manual feature engineering.

**Why DL over traditional ML?**
- Excels on unstructured data: images, audio, text, video
- Automatically learns features (vs manual feature engineering in classical ML)
- Scales better with very large datasets
- Requires more data and compute than classical ML for comparable structured-data tasks

---

## 13.2 The Artificial Neuron (Perceptron)

```
output = activation( Σ(weight_i × input_i) + bias )
```

A single neuron computes a weighted sum of its inputs, adds a bias, then passes the result through a non-linear **activation function**.

```python
import numpy as np
def neuron(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    return max(0, z)   # ReLU activation
```

---

## 13.3 Neural Network Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
```
- **Input layer**: receives raw features (one node per feature)
- **Hidden layer(s)**: intermediate layers that learn increasingly abstract representations — "deep" networks have multiple hidden layers
- **Output layer**: produces the final prediction (e.g., single node with sigmoid for binary classification, `n` nodes with softmax for multi-class)

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(20,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")     # binary classification output
])
```

---

## 13.4 Activation Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **Sigmoid** | `1/(1+e^-x)` | Output layer, binary classification (0-1 range) |
| **Tanh** | `(e^x - e^-x)/(e^x + e^-x)` | Hidden layers (zero-centered, -1 to 1) |
| **ReLU** | `max(0, x)` | Most common in hidden layers — fast, reduces vanishing gradient |
| **Leaky ReLU** | `x if x>0 else 0.01x` | Fixes "dying ReLU" problem |
| **Softmax** | normalizes to probabilities summing to 1 | Output layer, multi-class classification |

🔑 **Interview Q: Why is ReLU preferred over Sigmoid in hidden layers?**
Sigmoid saturates for large positive/negative inputs, causing gradients to shrink toward zero during backpropagation (**vanishing gradient problem**), slowing or stalling learning in deep networks. ReLU's gradient is constant (1) for positive inputs, avoiding this saturation and enabling faster, more stable training in deep architectures — though it can suffer from "dying ReLU" (neurons stuck outputting 0), which Leaky ReLU addresses.

---

## 13.5 Forward Propagation & Backpropagation

**Forward propagation**: input flows through the network layer by layer, producing a prediction.

**Backpropagation**: after computing the loss (error between prediction and true label), the algorithm computes gradients of the loss with respect to every weight in the network (using the **chain rule** of calculus), propagating error backward from the output layer to the input layer.

**Gradient Descent**: weights are updated in the direction that reduces loss:
```
weight_new = weight_old - learning_rate × gradient
```

- **Learning rate**: controls step size — too high overshoots the minimum (unstable), too low trains very slowly.
- **Epoch**: one full pass through the entire training dataset.
- **Batch size**: number of samples processed before updating weights (mini-batch gradient descent is standard).

---

## 13.6 Loss Functions

| Task | Loss Function |
|------|---------------|
| Binary classification | Binary Cross-Entropy |
| Multi-class classification | Categorical Cross-Entropy |
| Regression | MSE or MAE |

```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

---

## 13.7 Optimizers

| Optimizer | Key Idea |
|-----------|----------|
| **SGD** (Stochastic Gradient Descent) | Basic weight update using gradients from mini-batches |
| **Momentum** | Accumulates a moving average of past gradients to smooth updates and accelerate convergence |
| **RMSprop** | Adapts learning rate per parameter based on recent gradient magnitudes |
| **Adam** | Combines Momentum + RMSprop — adaptive learning rates per parameter; the most commonly used default optimizer |

---

## 13.8 Overfitting Prevention in Neural Networks

- **Dropout**: randomly "turns off" a fraction of neurons during each training step, preventing co-adaptation and forcing redundant, robust representations.
  ```python
  keras.layers.Dropout(0.3)   # drops 30% of neurons randomly each step
  ```
- **Early Stopping**: halt training once validation loss stops improving, before the model starts overfitting.
- **L1/L2 Regularization**: penalize large weights in the loss function.
- **Batch Normalization**: normalizes layer inputs, stabilizing and speeding up training.
- **Data Augmentation**: artificially expand training data (especially for images) via rotations, flips, crops, etc.

---

## 13.9 Convolutional Neural Networks (CNNs)

Specialized for **grid-like data** (images), using **convolutional filters** that slide across the input to detect local patterns (edges, textures, shapes) — patterns detected in early layers combine into more complex features in deeper layers.

**Key components:**
- **Convolution layer**: applies learnable filters/kernels to extract spatial features
- **Pooling layer** (Max/Average Pooling): downsamples feature maps, reducing dimensionality while retaining important information
- **Flatten + Dense layers**: convert extracted features into final classification/regression output

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
```

---

## 13.10 Recurrent Neural Networks (RNNs) & LSTMs

Designed for **sequential data** (text, time series, audio) — unlike standard feedforward networks, RNNs maintain a "hidden state" that carries information from previous time steps forward.

🔑 **Interview Q: What problem do standard RNNs have, and how do LSTMs fix it?**
Standard RNNs suffer from the **vanishing gradient problem** over long sequences — gradients shrink exponentially as they're backpropagated through many time steps, making it hard to learn long-range dependencies. **LSTM (Long Short-Term Memory)** networks solve this using a **gating mechanism** (input, forget, and output gates) and a separate **cell state** that allows information to flow largely unchanged across many time steps, preserving long-term dependencies.

```python
model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=64),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation="sigmoid")
])
```

---

## ✅ Chapter 13 Summary

- Neural networks stack layers of neurons, each applying weighted sums + non-linear activation
- ReLU is the standard hidden-layer activation, avoiding vanishing gradients better than Sigmoid/Tanh
- Backpropagation + gradient descent (via Adam optimizer, typically) trains the network by minimizing a loss function
- Dropout, early stopping, and regularization combat overfitting
- CNNs excel at spatial data (images) via convolution + pooling; RNNs/LSTMs excel at sequential data (text, time series) via memory of past states

---

[← Previous: Model Evaluation](12_model_evaluation.md) | [Back to Index](../README.md) | [Next: NLP →](14_nlp.md)
