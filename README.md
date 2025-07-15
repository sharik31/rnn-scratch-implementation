# RNN from Scratch using NumPy

A **vanilla Recurrent Neural Network (RNN)** implemented **from scratch using NumPy**, including:  

. Forward propagation  
. Backpropagation Through Time (BPTT)  
.Softmax output & cross-entropy loss  
.Parameter updates with gradient descent  
. Minimal training loop on a toy dataset  

This project is a simple educational implementation to understand how RNNs work internally without relying on deep learning frameworks like TensorFlow or PyTorch.  

---

## 🚀 Features  

- **Single-step RNN cell** with `tanh` activation  
- **Softmax output layer** for classification  
- **Cross-entropy loss**  
- **BPTT implementation** to compute gradients  
- **Tiny training loop** to verify learning works  

---

## 📂 Project Structure 
```text
rnn-from-scratch-numpy/
│
├── README.md                 # Project documentation
├── rnn_from_scratch.ipynb    # Notebook version (optional)
├── rnn_from_scratch.py       # Main RNN implementation
├── doc/                      # Handwritten implementation notes/images
├── requirements.txt          # NumPy dependency

```




---

## 📊 Algorithm & Architecture Overview  

This project implements a **Vanilla RNN** with the following architecture:  

- **Input size (`n_x`)** → 2  
- **Hidden units (`n_a`)** → 4  
- **Output size (`n_y`)** → 2  
- **Sequence length (`T`)** → 3 timesteps  
- **Batch size (`m`)** → 1  

---

### 🏗 **Architecture**

For each timestep **t**, the RNN cell computes:  

1️⃣ **Hidden state update**  

`a_t` = `tanh(Wax} x_t + Waa a_{t-1} + b_a)`


where:  
- `x_t` → input at time **t**  
- `a_{t-1}` → hidden state from previous timestep  
- `W_ax` → weights for input → hidden  
- `W_aa` → weights for hidden → hidden  
- `b_a` → bias for hidden state  

2️⃣ **Output prediction**  

`{y}_t` = `{softmax}(W_{ya} a_t + b_y)`


where:  
- `W_ya` → weights from hidden → output  
- `b_y` → bias for output  

---

### 🔄 **Forward Pass**  

- For **T=3** timesteps, the RNN runs sequentially:  
  - Takes `x_1`, computes `a_1`, predicts `y_1`  
  - Takes `x_2`, computes `a_2`, predicts `y_2`  
  - Takes `x_3`, computes `a_3`, predicts `y_3`  

The hidden state **flows through time** and carries memory of previous steps.

---

### 🔁 **Backward Pass (BPTT)**  

- Uses **Backpropagation Through Time**  
- Gradients are computed for each timestep  
- They are **accumulated** for all time steps before updating parameters  

---

### ⚙️ **Parameter Shapes**

- `W_ax` → (4, 2)   → hidden units × input size  
- `W_aa` → (4, 4)   → hidden units × hidden units  
- `W_ya` → (2, 4)   → output size × hidden units  
- `b_a`  → (4, 1)   → hidden bias  
- `b_y`  → (2, 1)   → output bias  

---

### 🧠 **Training Setup**

- **Optimizer:** Simple Gradient Descent  
- **Learning rate:** 0.1  
- **Epochs:** 10  
- **Dataset:** Tiny toy dataset (sequence length 3)  
- **Target class:** Always class `0` for all timesteps  

---

This minimal example is **educational** and shows how RNNs work internally **without any deep learning frameworks**.

---

## 🛠 Installation  

```bash
git clone https://github.com/sharik31/rnn-from-scratch-numpy.git
pip install -r requirements.txt



