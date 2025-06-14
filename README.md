# LLM From Scratch 🚀

## 🧱 Current Progress

### 1. Bigram Model
- 📓 Notebook: [`bigram`](bigram_model.ipynb)
- 🧠 A simple character-level bigram language model implemented from scratch both neural network and lookup table based.

### 2. N-gram Autoregressive Model (Character-level)
- 📓 Paper followed: [`Bengio et al. 2003`](extras/mlp_paper.pdf)
- 🔧 Multi-layer perceptron-based character-level neural net with character-wise embedding vectors.

### 3. Batch Normalization in N-gram Character Model
- 📓 Papers:
  - Paper followed: [`Bengio et al. 2003`](extras/mlp_paper.pdf)
  - [`BatchNorm original`](extras/batch_paper.pdf)
  - [`Rethinking “Batch” in BatchNorm`](extras/batch_problem.pdf)
- 🔬 Applied Batch Normalization to the n-gram character-level model:
  - Explored effects on **forward-pass activations** and **backward-pass gradients**.
  - Highlighted **pitfalls** when normalization statistics are improperly scaled or applied.
  - Analyzed internal dynamics and stability improvements during training.

### 4. Manual Backpropagation Through Previous MLP

- 📓 Notebook: [`Back Prop`](backprop.ipynb)  
  A manual implementation of backpropagation through a simple MLP, useful for understanding gradient flow and low-level training mechanics.

### 5. WaveNet-like Architecture

- 📓 Paper Referenced: [`WaveNet (DeepMind, 2016)`](TinyToGPT/extras/wavenet.pdf)

Built upon the previous MLP by deepening it into a **tree-like structure** inspired by the **WaveNet** architecture. While the original WaveNet achieves hierarchical feature extraction efficiently through **causal dilated convolutions**, which not implemented in this 

### 6. GPT 
- Repository : [`GPT`](https://github.com/Haseebasif7/GPT)

 GPT-style **decoder-only Transformer**

### 🔹 7. Byte Pair Encoding (BPE) Tokenizer

A custom Byte Pair Encoding tokenizer implemented **from scratch**, including:

- UTF-8 byte-level processing
- GPT-style **regex-based token splitting**
- Dynamic vocabulary building via merge rules

📂 **Files:**
- 🧠 Tokenizer Class: [`BPE.py`](TinyToGPT/BPE.py)  
- 🧪 Debug Notebook: [`tokenizer.ipynb`](TinyToGPT/tokenizer.ipynb)
