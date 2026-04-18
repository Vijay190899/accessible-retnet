# TASK: Recreate "Accessible RetNet" (124M Parameters)

## 1. Project Objective
Implement and train a Retentive Network (RetNet) that matches the performance metrics of the "Accessible RetNet" paper. 
- [cite_start]**Target Metric:** Validation Perplexity of ~15.2 on WikiText-103.
- [cite_start]**Efficiency Goal:** 2x memory reduction during recurrent inference compared to standard KV cache[cite: 321, 456].

## 2. Model Configuration (124M Scale)
[cite_start]Implement the following hyperparameters exactly[cite: 175, 434]:
| Component | Value |
| :--- | :--- |
| **Model Dimension (d)** | 768 |
| **Layers (L)** | 12 |
| **Heads (H)** | 6 |
| **Head Dimension (Q, K)** | 128 |
| **Head Dimension (V)** | 256 |
| **FFN Dimension (d_ff)** | 1,536 |
| **Vocabulary Size** | 50,257 (GPT-2 Tokenizer) |
| **Max Sequence Length** | 512 |
| **Activation** | GELU (FFN) and SiLU (Retention Gate) |

## 3. Core Implementation Requirements

### [cite_start]A. The Retention Mechanism [cite: 108, 127]
Implement three computational paradigms:
1. [cite_start]**Parallel (Training):** $Output = ((QK^T \odot D) / \sqrt{d_k})V$[cite: 110, 111].
   - [cite_start]**Decay Matrix (D):** $D_{ij} = \gamma^{i-j}$ if $i \ge j$, else 0[cite: 98].
   - [cite_start]**Normalization:** $\tilde{D}_{i,j} = D_{i,j} / \sqrt{\sum_{k=1}^i D_{ik}}$ to stabilize training[cite: 104].
2. [cite_start]**Recurrent (Inference):** - $s_t = \gamma s_{t-1} + k_t^T v_t$[cite: 117].
   - [cite_start]$o_t = q_t s_t$[cite: 118].
3. [cite_start]**Multi-Scale Decay:** Each of the 6 heads must use: $\gamma_h = 1 - 2^{-(5+h)}$[cite: 129].

### B. Positional Encoding
- [cite_start]Use **xPos** (Extrapolatable Position Encoding) with rotary transformations[cite: 144, 153].

### C. Layer Components
- [cite_start]**Normalization:** Use LayerNorm for training stability[cite: 187].
- [cite_start]**Output:** Concatenated head outputs must pass through GroupNorm and a gated linear unit (Swish/SiLU)[cite: 134, 137].
- [cite_start]**Weight Tying:** The Token Embedding and Language Model head should share weights[cite: 161].

## 4. Training Recipe
[cite_start]Follow these settings to reproduce the experimental results[cite: 196, 203, 208]:
- [cite_start]**Dataset:** WikiText-103 (103M tokens)[cite: 191].
- [cite_start]**Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.98$, Weight Decay=0.05)[cite: 196, 199].
- [cite_start]**Learning Rate:** $3 \times 10^{-4}$ with a **Warmup-Cosine schedule** (1k warmup, 50k total steps)[cite: 197, 203, 204].
- [cite_start]**Precision:** Mixed Precision (FP16)[cite: 209].
- [cite_start]**Stability:** Apply Gradient Clipping at 1.0[cite: 200].
- [cite_start]**Batching:** Effective batch size of 32 (Use accumulation steps if needed)[cite: 207, 208].

## 5. Evaluation & Generation
[cite_start]Implement a generation pipeline to evaluate the "Perplexity-Quality Mismatch"[cite: 325, 457].
- [cite_start]**Decoding Strategies:** Temperature sampling (0.5-1.2), Top-p (0.9), Repetition Penalty (1.3), and N-gram blocking (n=3)[cite: 301, 307].
- [cite_start]**Expected Outcome:** Low perplexity (15.2) but potential semantic drift or factual inaccuracies without alignment[cite: 325, 327].