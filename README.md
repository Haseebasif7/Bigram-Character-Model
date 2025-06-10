# LLM From Scratch 🚀

## 🧱 Current Progress

### 1. Bigram Model
- 📓 Notebook: [`bigram`](bigram_model.ipynb)
- 🧠 A simple character-level bigram language model implemented from scratch both neural network and lookup table based.

### 2. MLP-based Character Model
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

### 4. Manual BackPropogation Through Previous MLP
