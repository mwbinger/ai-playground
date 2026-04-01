# AI Fundamentals Curriculum

A historically-grounded learning path through the key ideas in AI, building from the simplest neural model to modern systems. Each module is meant to be implemented from scratch (pure numpy where possible) to build real understanding.

## 1. The Perceptron (Rosenblatt, 1958)

Single-neuron binary classifier. Implement the perceptron learning algorithm, prove to yourself it converges for linearly separable data, then hit the XOR wall that Minsky & Papert famously highlighted.

**Key implementations:**
- Perceptron class with predict and train methods (both ±1 and 0/1 label conventions)
- Train on 2D linearly separable data, visualize the decision boundary evolving over epochs
- Demonstrate failure on XOR
- Verify the convergence bound: at most (R/γ)² mistakes for data with margin γ and ‖x‖ ≤ R

**Key reading:** Rosenblatt (1958), Minsky & Papert — *Perceptrons* (1969)

## 2. Multi-Layer Perceptrons & Backpropagation (Rumelhart, Hinton & Williams, 1986)

The idea that ended the first AI winter. A hidden layer lets you learn non-linear decision boundaries, and backpropagation provides an efficient way to compute gradients through the chain rule.

**Key implementations:**
- MLP from scratch with numpy — forward pass, loss computation, and backprop via the chain rule
- Solve XOR with a small hidden layer
- Train on MNIST
- Experiment with different activations (sigmoid, tanh, ReLU) and observe training dynamics

**Key reading:** Rumelhart, Hinton & Williams — "Learning representations by back-propagating errors" (1986)

## 3. Hopfield Networks & Boltzmann Machines (Hopfield 1982, Hinton & Sejnowski 1985)

Energy-based models and associative memory. A different paradigm from feedforward networks — these models define an energy landscape and settle into stable states.

**Key implementations:**
- Hopfield network that stores and recalls binary patterns
- Explore storage capacity limits (≈ 0.14N patterns for N neurons)
- Restricted Boltzmann machine with contrastive divergence training

**Key reading:** Hopfield — "Neural networks and physical systems with emergent collective computational abilities" (1982), Hinton — "A Practical Guide to Training Restricted Boltzmann Machines" (2010)

## 4. Convolutional Neural Networks (LeCun 1989 → AlexNet 2012)

Exploiting spatial structure through weight sharing and local receptive fields. The architecture that dominated computer vision for a decade.

**Key implementations:**
- Small CNN from scratch or with minimal framework help
- Implement convolution, pooling, and backprop through convolutional layers
- Train on MNIST/CIFAR-10
- Understand the LeNet → AlexNet progression: what changed was scale, GPUs, ReLU, and dropout

**Key reading:** LeCun et al. — "Gradient-based learning applied to document recognition" (1998), Krizhevsky et al. — "ImageNet Classification with Deep Convolutional Neural Networks" (2012)

## 5. Recurrent Neural Networks & LSTMs (Elman 1990, Hochreiter & Schmidhuber 1997)

Sequence modeling before transformers. The challenge of learning long-range dependencies and the gating mechanisms that address it.

**Key implementations:**
- Vanilla RNN for character-level language modeling
- Observe vanishing/exploding gradients firsthand
- Implement an LSTM cell — understand forget, input, and output gates
- Compare vanilla RNN vs LSTM on sequences requiring long-range memory

**Key reading:** Hochreiter & Schmidhuber — "Long Short-Term Memory" (1997), Karpathy — "The Unreasonable Effectiveness of Recurrent Neural Networks" (2015, blog post)

## 6. Word Embeddings & Representation Learning (word2vec 2013, GloVe 2014)

The insight that meaning lives in geometry. Dense vector representations of words learned from co-occurrence patterns.

**Key implementations:**
- Skip-gram with negative sampling from scratch
- Explore analogy arithmetic (king − man + woman ≈ queen)
- Visualize embedding space with t-SNE or PCA
- Compare with GloVe (co-occurrence matrix factorization approach)

**Key reading:** Mikolov et al. — "Efficient Estimation of Word Representations in Vector Space" (2013), Pennington et al. — "GloVe: Global Vectors for Word Representation" (2014)

## 7. Sequence-to-Sequence & Attention (Sutskever 2014, Bahdanau 2015)

The encoder-decoder architecture and the original attention mechanism that let models selectively focus on relevant parts of the input.

**Key implementations:**
- Seq2seq model for a toy translation or date-formatting task
- First without attention — feel the information bottleneck
- Then add Bahdanau (additive) attention and observe the improvement
- Visualize attention weights to see what the model is "looking at"

**Key reading:** Sutskever et al. — "Sequence to Sequence Learning with Neural Networks" (2014), Bahdanau et al. — "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)

## 8. The Transformer (Vaswani et al., 2017)

The architecture that changed everything. This module deserves the most time.

**Key implementations:**
- Scaled dot-product attention from scratch
- Multi-head attention
- Positional encodings (sinusoidal)
- Full encoder-decoder transformer stack with layer norm and residual connections
- Train on a toy task (e.g., copy, reverse, or small translation)

**Key reading:** Vaswani et al. — "Attention Is All You Need" (2017), "The Annotated Transformer" (Harvard NLP, blog post), "The Illustrated Transformer" (Jay Alammar, blog post)

## 9. BERT & GPT (Devlin 2018, Radford 2018)

Two paradigms from the same architecture: bidirectional masked language modeling (BERT) vs. autoregressive left-to-right generation (GPT).

**Key implementations:**
- Small GPT-style autoregressive language model, trained on a text corpus
- Understand pretraining vs. fine-tuning
- Implement masked language modeling (BERT-style) for comparison
- Explore how model scale affects capability

**Key reading:** Devlin et al. — "BERT: Pre-training of Deep Bidirectional Transformers" (2018), Radford et al. — "Improving Language Understanding by Generative Pre-Training" (2018), Radford et al. — "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)

## 10. Scaling, RLHF & Modern Alignment (2020–present)

How raw language models become useful assistants. More conceptual, but a simple RLHF loop on a toy model is very illuminating.

**Key topics & implementations:**
- Scaling laws (Kaplan et al.) — understand the power-law relationships between compute, data, parameters, and loss
- Instruction tuning / supervised fine-tuning (SFT)
- Reward modeling and RLHF (Ouyang et al. 2022)
- Constitutional AI and preference optimization (DPO)
- Implement a simple RLHF loop on a small model

**Key reading:** Kaplan et al. — "Scaling Laws for Neural Language Models" (2020), Ouyang et al. — "Training language models to follow instructions with human feedback" (2022), Bai et al. — "Constitutional AI" (2022)

## 11. Mixture of Experts, Long Context & Inference-Time Techniques

Modern architectural choices and deployment tricks that define the current frontier.

**Key topics:**
- Sparse Mixture of Experts (MoE) — routing, load balancing, why sparsity helps scale
- Position encoding beyond sinusoidal: RoPE, ALiBi
- KV caching and efficient inference
- Speculative decoding
- Chain-of-thought prompting and inference-time compute scaling

**Key reading:** Fedus et al. — "Switch Transformers" (2021), Su et al. — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021), Leviathan et al. — "Fast Inference from Transformers via Speculative Decoding" (2023), Wei et al. — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)

---

## Pacing notes

- **Modules 1–3**: A few days each given a strong math/ML background
- **Modules 4–6**: ~1 week each
- **Modules 7–8**: 2+ weeks — this is where the deepest learning happens
- **Modules 9–11**: Blend implementation with paper reading, flexible pacing