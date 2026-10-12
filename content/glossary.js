// The glossary: B's 121 terms (verbatim, one corrected), A's definition-bank entries that B lacks, and MLP terms
// taken from the MLP lesson. moduleId is an A module id, or null when A has no module for the term yet.
(function () {
  window.AIO_GLOSSARY = [
    {
      "id": "activation-function",
      "term": "Activation Function",
      "aka": null,
      "definition": "A nonlinear function like ReLU, sigmoid, tanh, or GELU, applied elementwise to a layer’s outputs. Without activation, stacked layers collapse to a linear map making modelling complex patterns impossible. Rule of thumb: ReLU/GELU inside the network, sigmoid/softmax only at the output.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "activation",
        "nonlinearity"
      ]
    },
    {
      "id": "adam",
      "term": "Adam",
      "aka": "Adaptive Moment Estimation",
      "definition": "The default optimizer for deep learning. Combines momentum, scaled with a fixed parameter β, with per-parameter learning-rate scaling (exponential moving average of squared gradients). It works out of the box, though the learning rate still matters.",
      "moduleId": "optimization-adam-sgd",
      "tags": [
        "optimizer",
        "gradient-descent",
        "momentum"
      ]
    },
    {
      "id": "adamw",
      "term": "AdamW",
      "aka": null,
      "definition": "Adam with decoupled weight decay. In regular Adam, L2 regularization interacts weirdly with the LR-scaling term. AdamW applies weight decay as a separate step and is the default in modern transformer training.",
      "moduleId": "advanced-optimization",
      "tags": [
        "optimizer",
        "regularization",
        "weight-decay"
      ]
    },
    {
      "id": "adjusted-rand-index",
      "term": "Adjusted Rand index",
      "aka": "ARI",
      "definition": "Compares two partitions of the same points pair by pair, asking whether they agree on putting each pair together or apart, and corrects for the agreement chance alone would give. It is 1 for identical partitions, about 0 for random ones, and negative when worse than random. Landmine: plain accuracy against cluster IDs can score a perfect clustering at zero, because the IDs are arbitrary, and ARI never looks at them.",
      "moduleId": "clustering",
      "tags": [
        "clustering",
        "evaluation",
        "ari"
      ]
    },
    {
      "id": "advantage",
      "term": "Advantage",
      "aka": null,
      "definition": "How much better an action is than what the policy normally does in that state: A(s,a) = Q(s,a) - V(s). Subtracting V costs nothing in bias and collapses the variance, which is why raw returns are almost never used directly. Landmine: an advantage is defined against the current policy, so it goes stale the instant you update.",
      "moduleId": null,
      "tags": [
        "reinforcement-learning",
        "advantage",
        "variance-reduction"
      ]
    },
    {
      "id": "adversarial-example",
      "term": "Adversarial example",
      "aka": null,
      "definition": "An input perturbed by an amount too small to see that flips the model prediction, found by taking the gradient of the loss with respect to the INPUT and stepping along it. Landmine: an attack failing does not mean a model is robust, it often means the defence hid the gradient.",
      "moduleId": null,
      "tags": [
        "adversarial",
        "robustness",
        "fgsm",
        "input-gradients"
      ]
    },
    {
      "id": "alibi",
      "term": "ALiBi",
      "aka": "Attention with Linear Biases",
      "definition": "Adds -m(i - j) to the attention logit between query i and key j, with a fixed slope m per head and no learned parameters. Inside a softmax that is a multiplicative decay of e^(-m) per step of distance, which is why it carries over to linear attention as a decayed recurrence. Landmine: it works with a KV cache, since the bias for a new query is computed from offsets when the query arrives.",
      "moduleId": "relative-position-rope-positional-methods",
      "tags": [
        "transformers",
        "positional-encoding",
        "alibi"
      ]
    },
    {
      "id": "attention",
      "term": "Attention",
      "aka": null,
      "definition": "A learned soft lookup: for each query, take a weighted average of value vectors where the weights are softmax(query · key / √d_k). The core mechanism inside a Transformer, and the reason models can recall content across long contexts.",
      "moduleId": "transformer-fundamentals",
      "tags": [
        "attention",
        "transformers",
        "softmax"
      ]
    },
    {
      "id": "autoencoder",
      "term": "Autoencoder",
      "aka": null,
      "definition": "A network trained to reconstruct its input through a bottleneck. The bottleneck forces it to learn a compressed representation. VAEs add a probabilistic twist (see ELBO).",
      "moduleId": "representation-learning-autoencoders-vaes-latent-spaces",
      "tags": [
        "generative-models",
        "representation-learning"
      ]
    },
    {
      "id": "autograd",
      "term": "Autograd",
      "aka": null,
      "definition": "Automatic differentiation implemented by tracking every operation on tensors and replaying the reverse-mode chain rule on .backward(). PyTorch’s defining feature: no real model needs a hand-written gradient.",
      "moduleId": "pytorch-tensors-autograd",
      "tags": [
        "pytorch",
        "autodiff",
        "reverse-mode"
      ]
    },
    {
      "id": "backpropagation",
      "term": "Backpropagation",
      "aka": null,
      "definition": "The reverse-mode chain rule applied to a neural network’s computation graph. Each layer receives the gradient of the loss with respect to its output and computes gradients with respect to its inputs and parameters. Every deep-learning framework implements it under the hood.",
      "moduleId": "backpropagation-mechanics",
      "tags": [
        "backpropagation",
        "chain-rule",
        "gradients"
      ]
    },
    {
      "id": "bagging",
      "term": "Bagging",
      "aka": null,
      "definition": "Bootstrap AGGregatING: train many high-variance predictors (e.g. deep decision trees) on bootstrap resamples of the data and average their predictions. Reduces variance without changing bias. Random forests are bagging + random feature subsets.",
      "moduleId": "tree-based-models-ensembles",
      "tags": [
        "ensembles",
        "bagging",
        "random-forests",
        "variance-reduction"
      ]
    },
    {
      "id": "batch-normalization",
      "term": "Batch Normalization",
      "aka": "BatchNorm",
      "definition": "Normalize each activation dimension over the mini-batch (subtract mean, divide by std), then scale/shift by learned parameters. Speeds up training, but leaks batch statistics: hence LayerNorm is preferred in Transformers.",
      "moduleId": "neural-network-regularization-dropout-batchnorm",
      "tags": [
        "normalization",
        "batch-norm",
        "training-dynamics"
      ]
    },
    {
      "id": "bayes-theorem",
      "term": "Bayes’ Theorem",
      "aka": null,
      "definition": "Conditional Probability: P(A|B) = P(B|A) P(A) / P(B). Calculates the posterior probability (A given B) of a hypothesis given evidence. The foundation of Bayesian inference.",
      "moduleId": "probability-foundations",
      "tags": [
        "probability",
        "bayes",
        "conditional-probability"
      ]
    },
    {
      "id": "beam-search",
      "term": "Beam search",
      "aka": null,
      "definition": "Decoding that keeps the B most probable partial sequences by total log-probability at every step and extends all of them. It finds sequences greedy decoding misses when the best first token does not start the best sequence. Landmine: for open-ended text the most probable sequence is often dull and repetitive, which is why chat models sample instead.",
      "moduleId": "gpt-causal-lm",
      "tags": [
        "decoding",
        "generation",
        "language-models"
      ]
    },
    {
      "id": "bellman-equation",
      "term": "Bellman equation",
      "aka": null,
      "definition": "Express a state’s value as immediate reward plus the discounted value of what follows. The inner sum averages over next states; the outer sum averages over the policy’s actions. Replacing the outer average with a maximum gives the optimality equation.",
      "moduleId": "reinforcement-learning",
      "tags": [
        "reinforcement-learning"
      ]
    },
    {
      "id": "bias",
      "term": "Bias",
      "aka": null,
      "definition": "The number added to a neuron's weighted sum before the activation function. The bias sets the threshold: with a large negative bias the output stays near 0 unless the weighted inputs are large.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "parameters"
      ]
    },
    {
      "id": "bias-variance-tradeoff",
      "term": "Bias-Variance Tradeoff",
      "aka": null,
      "definition": "Expected test error decomposes into bias² (systematic error from a too-simple model) + variance (sensitivity to the training set) + irreducible noise. High-capacity models have low bias, high variance. Regularization trades one for the other.",
      "moduleId": "bias-variance-cross-validation",
      "tags": [
        "generalization",
        "overfitting",
        "regularization"
      ]
    },
    {
      "id": "bpe",
      "term": "BPE",
      "aka": "Byte-Pair Encoding",
      "definition": "A subword tokenizer built by iteratively merging the most frequent adjacent symbol pair. Compresses vocabulary while keeping rare words representable. Modern LLMs use byte-level BPE variants (e.g. GPT-2’s tokenizer).",
      "moduleId": "natural-language-processing-tokenization-embeddings",
      "tags": [
        "tokenization",
        "nlp",
        "subword"
      ]
    },
    {
      "id": "calibration",
      "term": "Calibration",
      "aka": null,
      "definition": "A classifier is calibrated when its predicted probabilities match empirical frequencies: of the samples predicted 0.8, about 80% are positive. Deep nets are usually overconfident. Temperature scaling is the standard cheap fix.",
      "moduleId": "model-evaluation",
      "tags": [
        "classification",
        "calibration",
        "temperature-scaling"
      ]
    },
    {
      "id": "catastrophic-cancellation",
      "term": "Catastrophic cancellation",
      "aka": null,
      "definition": "Subtracting two nearly equal large numbers destroys the leading digits, leaving an answer built from the least reliable ones. The classic victim is Var(x) = E[x^2] - E[x]^2, which returns exactly 0 for data of variance 1 sitting at an offset of 1e8, and a negative variance at 1e10. Landmine: it degrades smoothly rather than failing, losing about two digits per factor of 100 in the offset, so a test on centred data shows nothing wrong.",
      "moduleId": "programming-essentials",
      "tags": [
        "numerics",
        "floating-point",
        "variance"
      ]
    },
    {
      "id": "cauchy-schwarz-inequality",
      "term": "Cauchy–Schwarz inequality",
      "aka": null,
      "definition": "The absolute inner product is at most the product of the two vector lengths. Two vectors have the largest possible inner product, for their lengths, when they point along the same line. This is why the gradient gives the steepest direction among unit-length steps.",
      "moduleId": "calculus-gradients",
      "tags": [
        "linear-algebra",
        "inequality"
      ]
    },
    {
      "id": "causal-mask",
      "term": "Causal Mask",
      "aka": null,
      "definition": "A boolean/attention mask that zeroes out (or -∞-fills) attention weights from position i to positions j > i, so a token can only attend to the past. Turns bidirectional attention into an autoregressive model.",
      "moduleId": "gpt-causal-lm",
      "tags": [
        "attention",
        "autoregression",
        "transformers"
      ]
    },
    {
      "id": "cbow",
      "term": "CBOW",
      "aka": "Continuous Bag of Words",
      "definition": "The other word2vec architecture: average the context embeddings and predict the centre word, the mirror of skip-gram. Roughly 2W times cheaper, since it makes one training example per position rather than one per centre-context pair. Landmine: the correct gradient divides each context word update by 2W, and word2vec.c, Gensim and fastText all left that division out. With the correct update CBOW matches skip-gram (Irsoy et al., 2021), so the folklore that skip-gram wins on rare words owes more to that bug than to the method.",
      "moduleId": "natural-language-processing-tokenization-embeddings",
      "tags": [
        "nlp",
        "embeddings",
        "word2vec"
      ]
    },
    {
      "id": "central-limit-theorem",
      "term": "Central limit theorem",
      "aka": "CLT",
      "definition": "A normalized average of many independent observations approaches a normal distribution.",
      "moduleId": "statistics-hypothesis-testing",
      "tags": [
        "probability",
        "statistics"
      ]
    },
    {
      "id": "chain-rule",
      "term": "Chain rule",
      "aka": null,
      "definition": "Differentiate a composition by multiplying its local derivatives. A change in $x$ first changes $f$, which then changes $g$. Multiply the two sensitivities. When the influence travels along several paths, add their contributions; backpropagation organizes this calculation for a network.",
      "moduleId": "calculus-gradients",
      "tags": [
        "calculus",
        "backpropagation"
      ]
    },
    {
      "id": "condition-number",
      "term": "Condition Number",
      "aka": null,
      "definition": "A measure of how sensitive the output of a function is to changes in the input. A high condition number indicates that small changes in the input can lead to large changes in the output (lower is better).",
      "moduleId": "advanced-optimization",
      "tags": [
        "linear-algebra",
        "condition-number",
        "optimization"
      ]
    },
    {
      "id": "contraction-mapping",
      "term": "Contraction mapping",
      "aka": "Banach fixed-point theorem",
      "definition": "A map that shrinks distances by a uniform factor smaller than one. On a nonempty complete metric space, a contraction mapping the space into itself has one fixed point, and repeated application converges to it. Discounted Bellman updates use this result to justify value iteration.",
      "moduleId": "reinforcement-learning",
      "tags": [
        "reinforcement-learning",
        "analysis"
      ]
    },
    {
      "id": "convex-function",
      "term": "Convex Function",
      "aka": null,
      "definition": "f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for all λ in [0,1]. Local minima is the global minima. MSE, logistic loss, and hinge loss are convex. Neural-net losses are not.",
      "moduleId": "advanced-optimization",
      "tags": [
        "optimization",
        "convexity"
      ]
    },
    {
      "id": "convolution",
      "term": "Convolution",
      "aka": null,
      "definition": "A matrix of weights (kernels) applied to an image patch, allowing for local feature detection. The same feature detector sees every location, allowing for translation equivariance.",
      "moduleId": "convolutional-neural-networks",
      "tags": [
        "cnn",
        "convolution",
        "translation-equivariance"
      ]
    },
    {
      "id": "correlation",
      "term": "Correlation",
      "aka": null,
      "definition": "Covariance divided by both standard deviations, landing in [-1, 1] and free of units. Landmine: it measures LINEAR association only. For y = x^2 with symmetric x the correlation is essentially zero while x determines y completely, so a correlation of 0 supports no conclusion about independence.",
      "moduleId": "expectations-multivariate-distributions",
      "tags": [
        "probability",
        "statistics"
      ]
    },
    {
      "id": "cosine-similarity",
      "term": "Cosine Similarity",
      "aka": null,
      "definition": "cos(θ) = u·v / (‖u‖ ‖v‖). Measures alignment (ignoring magnitude) of two vectors. The standard similarity metric for word/sentence embeddings.",
      "moduleId": "natural-language-processing-tokenization-embeddings",
      "tags": [
        "embeddings",
        "similarity",
        "nlp"
      ]
    },
    {
      "id": "covariance",
      "term": "Covariance",
      "aka": null,
      "definition": "How two variables move together: Cov(X,Y) = E[(X-EX)(Y-EY)], with Var(X) = Cov(X,X) as the special case. Stacked into a matrix it is symmetric and positive semidefinite, which is what lets PCA use an eigendecomposition, and its trace equals the sum of its eigenvalues, so total variance is conserved and PCA only redistributes it. Landmine: the units are the product of both variables, so the magnitude means nothing on its own.",
      "moduleId": "expectations-multivariate-distributions",
      "tags": [
        "probability",
        "statistics",
        "pca"
      ]
    },
    {
      "id": "cross-entropy-loss",
      "term": "Cross-entropy loss",
      "aka": null,
      "definition": "The KL divergence between the label distribution and the model’s predicted distribution, plus a constant, the entropy of the labels. For one-hot labels, it’s -log(p_correct). The right loss for classification: gradients scale with 1 − p_correct, so confident wrong predictions are penalized heavily.",
      "moduleId": "information-theory-core",
      "tags": [
        "loss-functions",
        "classification",
        "softmax"
      ]
    },
    {
      "id": "cross-validation",
      "term": "Cross-Validation",
      "aka": "CV",
      "definition": "Estimate generalization by rotating a slice of the data as the validation set. k-fold: split into k folds, train k times, take average. Low-cost preventative measure to picking a lucky train/test split.",
      "moduleId": "bias-variance-cross-validation",
      "tags": [
        "validation",
        "model-selection",
        "k-fold"
      ]
    },
    {
      "id": "data-leakage",
      "term": "Data leakage",
      "aka": null,
      "definition": "Any path by which information from outside the training fold reaches the model or its preprocessing. Labels are one such path rather than the definition. Fitting a scaler, imputer or feature selector on all rows leaks, as does a target encoding computed over the full dataset, and so does shuffling a time series. Landmine: leakage essentially always flatters, so it reports good news you cannot cash and misdirects the hours you have left.",
      "moduleId": "data-processing-features",
      "tags": [
        "evaluation",
        "leakage",
        "pipelines"
      ]
    },
    {
      "id": "dead-relu",
      "term": "Dead ReLU",
      "aka": null,
      "definition": "A unit whose weighted sum is negative for every input outputs 0, and since the slope there is 0 it gets no gradient, so it stays off for good. Leaky ReLU, $\\max(0.01z, z)$, keeps a small slope on the negative side for that reason.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "activation",
        "relu",
        "training"
      ]
    },
    {
      "id": "determinant",
      "term": "Determinant",
      "aka": null,
      "definition": "The factor by which a square matrix scales volume, negative when it flips orientation, and zero exactly when the matrix is singular. It equals the product of the eigenvalues, just as the trace equals their sum.",
      "moduleId": "linear-algebra-i-vectors-matrices-systems-transformations",
      "tags": [
        "linear-algebra",
        "eigenvalues"
      ]
    },
    {
      "id": "diffusion-model",
      "term": "Diffusion model",
      "aka": null,
      "definition": "A generative model that learns to denoise noisy data generated through the forward process. Forward process gradually adds Gaussian noise. The model learns to reverse it at random time steps. Stable Diffusion, DALL·E 2, Sora.",
      "moduleId": "generative-adding-noise",
      "tags": [
        "diffusion",
        "generative-models",
        "denoising",
        "ddpm",
        "stable-diffusion"
      ]
    },
    {
      "id": "dropout",
      "term": "Dropout",
      "aka": null,
      "definition": "During training, randomly zero out each activation with probability p, and scale surviving activations by 1/(1-p) (inverted dropout). At test time, no dropout. Prevents model from relying on specific neurons.",
      "moduleId": "neural-network-regularization-dropout-batchnorm",
      "tags": [
        "regularization",
        "dropout"
      ]
    },
    {
      "id": "early-stopping",
      "term": "Early Stopping",
      "aka": null,
      "definition": "Stop training when validation loss stops improving for an extended period. An implicit regularizer at no cost. Save the best checkpoint so far, not the latest one.",
      "moduleId": "optimization-adam-sgd",
      "tags": [
        "regularization",
        "training",
        "validation"
      ]
    },
    {
      "id": "eigenvalue-eigenvector",
      "term": "Eigenvalue / Eigenvector",
      "aka": null,
      "definition": "For a square matrix A, a nonzero vector v with Av = λv. Eigenvectors are directions A only stretches (never rotates). PCA finds the top eigenvectors of the covariance. The spectral theorem is the SVD’s square-symmetric special case.",
      "moduleId": "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
      "tags": [
        "linear-algebra",
        "eigendecomposition",
        "pca"
      ]
    },
    {
      "id": "elbo",
      "term": "ELBO",
      "aka": "Evidence Lower BOund",
      "definition": "The VAE training objective: a tractable lower bound on log p(x), equal to a reconstruction term minus KL(q(z|x) ‖ p(z)). Maximizing it approximately maximizes the likelihood.",
      "moduleId": "vae-encoder-decoder-generation",
      "tags": [
        "vae",
        "variational-inference",
        "kl-divergence"
      ]
    },
    {
      "id": "embedding",
      "term": "Embedding",
      "aka": null,
      "definition": "A learned map from discrete IDs (tokens, users, items) to dense vectors. Similarity in the vector space captures semantic similarity. The lookup table is nn.Embedding in PyTorch.",
      "moduleId": "natural-language-processing-tokenization-embeddings",
      "tags": [
        "embeddings",
        "representation-learning"
      ]
    },
    {
      "id": "ensemble",
      "term": "Ensemble",
      "aka": null,
      "definition": "A prediction combining multiple (usually low-complexity) models (average, majority vote, stacking). Almost always beats a single model, bagging reduces variance, boosting reduces bias. The logic is similar to the \"wisdom of the crowd:\" many weak predictors can combine to a strong predictor.",
      "moduleId": "tree-based-models-ensembles",
      "tags": [
        "ensembles",
        "bagging",
        "boosting"
      ]
    },
    {
      "id": "entropy",
      "term": "Entropy",
      "aka": null,
      "definition": "Average surprise of a distribution, H(p) = -sum p log p, maximal for uniform and zero for deterministic. It is the floor on the cost of encoding samples from p, which is why cross-entropy can never beat it. Landmine: entropy is a property of the distribution, not of any one sample.",
      "moduleId": "information-theory-core",
      "tags": [
        "information-theory",
        "entropy",
        "cross-entropy"
      ]
    },
    {
      "id": "epoch",
      "term": "Epoch",
      "aka": null,
      "definition": "One full pass over the training set. Modern deep learning counts \"steps\" or \"tokens seen\" instead. Large models rarely finish even one epoch of their pretraining data.",
      "moduleId": null,
      "tags": [
        "training",
        "terminology"
      ]
    },
    {
      "id": "expected-value",
      "term": "Expected value",
      "aka": null,
      "definition": "A probability-weighted average of a random quantity. For a continuous variable, replace the sum with an integral against its density. An expectation need not be a possible outcome: the expected roll of a fair die is 3.5. Linearity lets us add expectations without assuming independence.",
      "moduleId": "expectations-multivariate-distributions",
      "tags": [
        "probability"
      ]
    },
    {
      "id": "exposure-bias",
      "term": "Exposure bias",
      "aka": null,
      "definition": "The mismatch between training with teacher forcing, where every position sees the true prefix, and generation, where the model conditions on its own earlier outputs. An early mistake puts the model somewhere it never trained, and errors can compound.",
      "moduleId": "sequence-models-transfer-learning",
      "tags": [
        "language-models",
        "teacher-forcing",
        "generation"
      ]
    },
    {
      "id": "fine-tuning",
      "term": "Fine-tuning",
      "aka": null,
      "definition": "Train a pretrained model further on task-specific data. Full fine-tuning updates all parameters. Parameter-efficient methods (LoRA, adapters) update only a small subset (freezing the other parameters).",
      "moduleId": "sequence-models-transfer-learning",
      "tags": [
        "transfer-learning",
        "fine-tuning",
        "pretraining"
      ]
    },
    {
      "id": "flashattention",
      "term": "FlashAttention",
      "aka": null,
      "definition": "An exact attention implementation that never writes the n-by-n score matrix to HBM, tiling the computation in SRAM and using the online-softmax rescaling identity. Landmine: it does not reduce FLOPs and is not an approximation, it reduces memory traffic, which is what actually costs you.",
      "moduleId": "efficient-attention-long-context",
      "tags": [
        "attention",
        "flash-attention",
        "memory",
        "online-softmax"
      ]
    },
    {
      "id": "focal-loss",
      "term": "Focal loss",
      "aka": null,
      "definition": "Cross-entropy scaled by (1 - p_t)^gamma, so confidently-correct examples contribute almost nothing. At gamma = 2 an anchor already correct at p_t = 0.99 has its loss cut 10,000-fold while a hard case at 0.25 is cut only twofold. It exists because a one-stage detector with a feature pyramid produces hundreds of thousands of anchors per image, almost all background, which plain cross-entropy lets drown the real objects.",
      "moduleId": "object-detection-fpn-yolo",
      "tags": [
        "detection",
        "loss",
        "class-imbalance"
      ]
    },
    {
      "id": "forward-pass",
      "term": "Forward pass",
      "aka": null,
      "definition": "Computing a network's output from its input, such as the 10 digit scores from an image's 784 pixels. In PyTorch it is the model's forward method, and the backward pass is derived from it by autograd.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "pytorch"
      ]
    },
    {
      "id": "fpn",
      "term": "FPN",
      "aka": "Feature Pyramid Network",
      "definition": "Sends semantics back down the resolution ladder: enlarge the coarser feature map 2 times, add the same-size backbone map through a 1x1 lateral convolution, repeat. Every pyramid level ends up both high-resolution and semantically strong, and all levels share a channel count (256 in the paper) so a single detection head with one set of weights serves them all.",
      "moduleId": "object-detection-fpn-yolo",
      "tags": [
        "detection",
        "multi-scale",
        "architecture"
      ]
    },
    {
      "id": "gan",
      "term": "GAN",
      "aka": "Generative Adversarial Network",
      "definition": "Two networks in a minimax game: a generator generates samples mimicking the real data distribution (rewarded for fooling discriminator), a discriminator predicts real vs. fake (rewarded for being correct). Trains generative models without an explicit likelihood, but notoriously unstable to train.",
      "moduleId": "generative-adversarial-networks",
      "tags": [
        "gan",
        "generative-models",
        "minimax"
      ]
    },
    {
      "id": "gcn",
      "term": "GCN",
      "aka": "Graph Convolutional Network",
      "definition": "A layer that averages a node’s neighbors’ features (weighted by degree), then applies a linear map and nonlinearity (activation functions). Permutation-equivariant: the output does not depend on node ordering.",
      "moduleId": "graph-neural-networks",
      "tags": [
        "gnn",
        "graph-neural-networks",
        "message-passing"
      ]
    },
    {
      "id": "glove",
      "term": "GloVe",
      "aka": "Global Vectors",
      "definition": "Word embeddings fitted so that a word vector dotted with a context vector, plus two biases, approximates the log of the co-occurrence count, by least squares weighted with f(x) = min((x/100)^0.75, 1). Without the weights and biases it is exactly a truncated SVD of the log-count matrix. The published vectors are the sum of the word and context vectors.",
      "moduleId": "natural-language-processing-tokenization-embeddings",
      "tags": [
        "nlp",
        "embeddings",
        "glove"
      ]
    },
    {
      "id": "gqa",
      "term": "GQA",
      "aka": "Grouped-Query Attention",
      "definition": "Grouped-query attention: several query heads share one key/value head, shrinking the KV cache by the group factor. Sits between full multi-head and multi-query, and an existing MHA checkpoint can be uptrained into it cheaply. Landmine: it shrinks the cache, not the number of query heads.",
      "moduleId": "efficient-sharing-keys-and-values",
      "tags": [
        "attention",
        "gqa",
        "kv-cache",
        "inference"
      ]
    },
    {
      "id": "gradient",
      "term": "Gradient",
      "aka": null,
      "definition": "The vector of partial derivatives of a scalar function. Each component tells you what a small change in that coordinate does to the output. Together they predict $f(x+\\Delta x)-f(x)\\approx\\nabla f(x)^\\top\\Delta x$. In Euclidean space, a nonzero gradient points toward the fastest local increase.",
      "moduleId": "calculus-gradients",
      "tags": [
        "calculus",
        "gradient"
      ]
    },
    {
      "id": "gradient-descent",
      "term": "Gradient Descent",
      "aka": null,
      "definition": "Take a step opposite the gradient: x ← x - η∇f(x). Converges linearly for convex quadratics at rate depending on the condition number. The foundation for every neural-net optimizer.",
      "moduleId": "optimization-adam-sgd",
      "tags": [
        "optimization",
        "gradient-descent"
      ]
    },
    {
      "id": "hessian",
      "term": "Hessian",
      "aka": null,
      "definition": "The matrix of second partial derivatives ∂²f/∂x_i∂x_j. Positive definite Hessian everywhere ⇔ f is strictly convex. Newton’s method uses it, but computing it for a big model is prohibitive.",
      "moduleId": "calculus-gradients",
      "tags": [
        "calculus",
        "optimization",
        "convexity"
      ]
    },
    {
      "id": "hidden-layer",
      "term": "Hidden layer",
      "aka": null,
      "definition": "A layer that is neither the input nor the output. The digit network in the MLP module has 784 inputs, two hidden layers of 16 neurons, and 10 outputs, one per digit.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "layers"
      ]
    },
    {
      "id": "hinge-loss",
      "term": "Hinge Loss",
      "aka": null,
      "definition": "max(0, 1 - y·f(x)): the SVM loss. Penalizes points inside the margin. Convex in f. Its subgradient gives the SVM update.",
      "moduleId": "kernel-methods-svms",
      "tags": [
        "svm",
        "loss-functions",
        "convex-optimization"
      ]
    },
    {
      "id": "hungarian-algorithm",
      "term": "Hungarian algorithm",
      "aka": null,
      "definition": "Solves the assignment problem exactly: given a table of scores, it finds the one-to-one matching of rows to columns with the best total, in polynomial time. SciPy ships it as linear_sum_assignment. It matches cluster IDs to classes before scoring, and predicted boxes to ground truth in DETR-style detectors.",
      "moduleId": null,
      "tags": [
        "clustering",
        "evaluation",
        "optimization"
      ]
    },
    {
      "id": "independent-and-identically-distributed",
      "term": "Independent and identically distributed",
      "aka": "i.i.d.",
      "definition": "Observations are independent draws from the same probability distribution. Independence means observing one draw does not change the distribution of another. “Identically distributed” means each draw has the same distribution. Repeated measurements from one person or consecutive video frames often violate independence.",
      "moduleId": "probability-foundations",
      "tags": [
        "probability",
        "statistics"
      ]
    },
    {
      "id": "infonce",
      "term": "InfoNCE",
      "aka": null,
      "definition": "The contrastive loss: cross-entropy where the classes are the other items in the batch, so the labels come free from pairing rather than annotation. It lower-bounds mutual information by log N minus the loss, which is why contrastive methods want enormous batches. Landmine: chance is log N, so a loss is only interpretable alongside its batch size.",
      "moduleId": "contrastive-learning",
      "tags": [
        "contrastive",
        "infonce",
        "clip",
        "mutual-information"
      ]
    },
    {
      "id": "initialization",
      "term": "Initialization",
      "aka": null,
      "definition": "The process of setting the initial values of the parameters in a machine learning model. Good initialization can lead to faster convergence and better performance. Common strategies include Xavier/Glorot initialization (for sigmoid and tanh activations) and He initialization (for ReLU networks).",
      "moduleId": null,
      "tags": [
        "initialization",
        "neural-networks",
        "optimization"
      ]
    },
    {
      "id": "iou",
      "term": "IoU",
      "aka": "Intersection over Union",
      "definition": "For two bounding boxes, area(A ∩ B) / area(A ∪ B) ∈ [0, 1]. The standard object-detection metric. Threshold 0.5 for \"correct detection\", 0.75 for stricter.",
      "moduleId": "object-detection-fpn-yolo",
      "tags": [
        "object-detection",
        "metrics",
        "bounding-boxes"
      ]
    },
    {
      "id": "jacobian",
      "term": "Jacobian",
      "aka": null,
      "definition": "A matrix recording how each output changes with each input. Row $i$ describes output $i$; column $j$ describes input $j$. Multiplying $J\\Delta x$ predicts the output change caused by a small input change.",
      "moduleId": "calculus-gradients",
      "tags": [
        "calculus",
        "matrix"
      ]
    },
    {
      "id": "jensen-inequality",
      "term": "Jensen inequality",
      "aka": null,
      "definition": "For a convex f and weights summing to 1, f(E[X]) <= E[f(X)]. The convexity definition extended from two points to any number. Equality holds only when f is affine or every input is equal, so observing the bound met tells you the inputs were all equal. Three results in this guide are one line of Jensen each: AM-GM from log, Gibbs (KL >= 0) from -log, and the ELBO.",
      "moduleId": "advanced-optimization",
      "tags": [
        "optimization",
        "convexity",
        "inequalities"
      ]
    },
    {
      "id": "k-fold-cross-validation",
      "term": "k-fold Cross-Validation",
      "aka": null,
      "definition": "Cross-validation (CV) with k folds. k=5 or 10 is standard. k=n is leave-one-subject-out (LOSO) CV, unbiased but high-variance and slow.",
      "moduleId": "bias-variance-cross-validation",
      "tags": [
        "cross-validation",
        "k-fold"
      ]
    },
    {
      "id": "k-means",
      "term": "k-means",
      "aka": null,
      "definition": "Cluster points by alternating: assign each point to the nearest centroid, then move each centroid to the mean of its assigned points. Lloyd’s algorithm. Monotone descent, no guarantee of global optimum: use k-means++ initialization.",
      "moduleId": "clustering",
      "tags": [
        "clustering",
        "k-means",
        "unsupervised-learning"
      ]
    },
    {
      "id": "kernel-trick",
      "term": "Kernel trick",
      "aka": "Mercer condition",
      "definition": "Replace every inner product with k(x, y) = phi(x).phi(y), computing a dot product in a high-dimensional space without ever forming phi. For a degree-2 kernel at d = 1000, phi would need 501,501 dimensions against the kernel cost of 1,000. Landmine: not every similarity function is a kernel. Mercer requires the Gram matrix be positive semidefinite for every sample, and a tanh pseudo-kernel can have a smallest eigenvalue of -54, meaning no feature space exists behind it and the dual stops being convex.",
      "moduleId": "kernel-methods-svms",
      "tags": [
        "svm",
        "kernels",
        "mercer"
      ]
    },
    {
      "id": "kl-divergence",
      "term": "KL divergence",
      "aka": "Kullback-Leibler",
      "definition": "KL(p‖q) = Σ p log(p/q). \"How much information is lost using q to code samples from p.\" Not symmetric, not a metric. Cross-entropy = H(p) + KL(p‖q).",
      "moduleId": "information-theory-core",
      "tags": [
        "probability",
        "information-theory",
        "divergence"
      ]
    },
    {
      "id": "knn",
      "term": "kNN",
      "aka": "k-Nearest Neighbors",
      "definition": "Predict the majority label (or mean value) of the k closest training points. Zero training time, expensive at inference. Curse of dimensionality is brutal past ~20 features.",
      "moduleId": "decision-trees-knn",
      "tags": [
        "knn",
        "classification",
        "lazy-learning"
      ]
    },
    {
      "id": "kv-cache",
      "term": "KV cache",
      "aka": null,
      "definition": "The stored keys and values for every past token, so autoregressive decoding does not recompute them each step. Causal masking is what makes this valid: a key computed at step t cannot change later. Landmine: it grows linearly with both context length and batch size, and decoding is memory-bound, so cache size is latency.",
      "moduleId": "efficient-attention-long-context",
      "tags": [
        "inference",
        "kv-cache",
        "attention",
        "memory"
      ]
    },
    {
      "id": "label-propagation",
      "term": "Label propagation",
      "aka": null,
      "definition": "Semi-supervised learning on a similarity graph: repeatedly replace every point with the weighted average of the label distributions of its neighbours, then reset the labeled points. The fixed point has a closed form, and its entry for class c is the probability that a random walk from the point reaches a class-c label first. Landmine: on images, raw-pixel distances make a poor graph, so propagate over a learned representation.",
      "moduleId": null,
      "tags": [
        "semi-supervised",
        "graphs",
        "label-propagation"
      ]
    },
    {
      "id": "lasso",
      "term": "Lasso",
      "aka": null,
      "definition": "Least squares with an L1 penalty, lambda times the sum of absolute weights. The corner of |w| at zero sets weights exactly to zero, so it selects features as it fits, and on a single coordinate it is soft-thresholding, sign(z) max(|z| - lambda, 0). Landmine: among correlated features it keeps one more or less arbitrarily, and scikit-learn scales its penalty differently from Ridge.",
      "moduleId": "linear-logistic-regression",
      "tags": [
        "regression",
        "regularization",
        "l1",
        "sparsity"
      ]
    },
    {
      "id": "law-of-large-numbers",
      "term": "Law of large numbers",
      "aka": null,
      "definition": "Under suitable assumptions, a sample average approaches the expected value. For independent, identically distributed variables with finite absolute expectation, the sample mean converges almost surely to $\\mathbb{E}X$. More samples stabilize the average; they do not make individual observations less variable.",
      "moduleId": "statistics-hypothesis-testing",
      "tags": [
        "probability",
        "statistics"
      ]
    },
    {
      "id": "law-of-total-expectation",
      "term": "Law of total expectation",
      "aka": "tower property",
      "definition": "Average a conditional expectation to recover the overall expectation. You can average within groups first and then average those group means, weighted by how often the groups occur. Bellman equations use this to separate the choice of action from the next-state transition.",
      "moduleId": "expectations-multivariate-distributions",
      "tags": [
        "probability"
      ]
    },
    {
      "id": "layer",
      "term": "Layer",
      "aka": null,
      "definition": "A set of neurons that read the same inputs. Put the weights in the rows of a matrix $W$ and the biases in a vector $\\mathbf{b}$, and $\\mathbf{a} = \\sigma(W\\mathbf{x} + \\mathbf{b})$ is all the outputs at once.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "layers"
      ]
    },
    {
      "id": "layer-normalization",
      "term": "Layer normalization",
      "aka": "LayerNorm",
      "definition": "Normalize each token’s features (mean/std over the feature dim), then scale/shift by learned parameters. Doesn’t depend on batch statistics: the reason Transformers use it, not BatchNorm.",
      "moduleId": "neural-network-regularization-dropout-batchnorm",
      "tags": [
        "normalization",
        "layernorm",
        "transformers"
      ]
    },
    {
      "id": "learning-rate",
      "term": "Learning rate",
      "aka": null,
      "definition": "The step size η in gradient descent. Too small: slow, may not escape saddle regions. Too big: divergence. Schedules that warm up then decay are the modern default.",
      "moduleId": "optimization-adam-sgd",
      "tags": [
        "optimization",
        "hyperparameters"
      ]
    },
    {
      "id": "least-squares",
      "term": "Least squares",
      "aka": null,
      "definition": "Minimize ‖Aᴅ - b‖². Closed form: ᴅ = (AᵀA)⁻¹Aᵀb (normal equations). Numerically use lstsq (QR/SVD) instead, forming AᵀA squares the condition number.",
      "moduleId": "linear-algebra-i-vectors-matrices-systems-transformations",
      "tags": [
        "linear-algebra",
        "regression",
        "ols"
      ]
    },
    {
      "id": "log-sum-exp",
      "term": "Log-sum-exp",
      "aka": "LSE",
      "definition": "log(Σ exp(x_i)). Numerically stable via the \"max trick\": LSE(x) = max(x) + log(Σ exp(x_i - max(x))). Its gradient is softmax(x).",
      "moduleId": "calculus-gradients",
      "tags": [
        "numerical-stability",
        "softmax"
      ]
    },
    {
      "id": "logistic-regression",
      "term": "Logistic regression",
      "aka": null,
      "definition": "A linear model with a sigmoid output, trained by minimizing binary cross-entropy. The negative log-likelihood is convex, so gradient descent finds the global minimum. The gateway ML model.",
      "moduleId": "linear-logistic-regression",
      "tags": [
        "classification",
        "logistic-regression",
        "convex-optimization"
      ]
    },
    {
      "id": "logit",
      "term": "Logit",
      "aka": "score",
      "definition": "An unnormalized score before sigmoid or softmax; in binary probability, the log-odds. Sigmoid turns a binary logit back into a probability. In multiclass models, “logits” means the scores passed to softmax; their differences determine the class probability ratios.",
      "moduleId": "linear-logistic-regression",
      "tags": [
        "classification",
        "softmax",
        "sigmoid"
      ]
    },
    {
      "id": "lora",
      "term": "LoRA",
      "aka": "Low-Rank Adaptation",
      "definition": "A fine-tuning method: freeze the pretrained weights W and learn a low-rank update BA (with rank r much less than d). Tiny extra params, big quality wins vs. full fine-tuning at a fraction of the memory.",
      "moduleId": null,
      "tags": [
        "fine-tuning",
        "peft",
        "lora"
      ]
    },
    {
      "id": "margin",
      "term": "Margin",
      "aka": null,
      "definition": "The distance from a decision boundary to the nearest training point. SVMs maximize the margin. Bigger margin → better generalization (structural risk minimization).",
      "moduleId": "kernel-methods-svms",
      "tags": [
        "svm",
        "generalization",
        "geometry"
      ]
    },
    {
      "id": "markov-property",
      "term": "Markov property",
      "aka": null,
      "definition": "Given the current state, earlier history provides no extra information about the next state.",
      "moduleId": "reinforcement-learning",
      "tags": [
        "reinforcement-learning",
        "probability"
      ]
    },
    {
      "id": "mixture-of-experts",
      "term": "Mixture of Experts",
      "aka": "MoE",
      "definition": "Replaces one FFN with N experts plus a router that sends each token to the top k, so parameters scale with N while per-token FLOPs scale with k. Landmine: memory scales with TOTAL parameters even though compute scales with active ones, and routing collapses onto a few experts unless it is explicitly balanced.",
      "moduleId": null,
      "tags": [
        "moe",
        "sparsity",
        "routing",
        "scaling"
      ]
    },
    {
      "id": "mle",
      "term": "MLE",
      "aka": "Maximum Likelihood Estimation",
      "definition": "Pick parameters that maximize the probability of the observed data: θ* = argmax Π p(x_i, θ). Cross-entropy loss is MLE for a categorical distribution. MSE is MLE for Gaussian noise.",
      "moduleId": "statistics-hypothesis-testing",
      "tags": [
        "statistics",
        "estimation",
        "likelihood"
      ]
    },
    {
      "id": "mlp",
      "term": "MLP",
      "aka": "Multi-Layer Perceptron",
      "definition": "A stack of fully-connected layers with nonlinearities in between. Universal function approximator, in principle. The classic \"vanilla neural network\".",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "mlp",
        "neural-networks"
      ]
    },
    {
      "id": "momentum",
      "term": "Momentum",
      "aka": null,
      "definition": "Add a moving average of past gradients to the update: v ← βv + g, x ← x - ηv. Damps oscillations in narrow valleys and accelerates in flat directions. Baked into Adam.",
      "moduleId": "optimization-adam-sgd",
      "tags": [
        "optimization",
        "momentum"
      ]
    },
    {
      "id": "multi-head-attention",
      "term": "Multi-head attention",
      "aka": null,
      "definition": "Split Q/K/V into h heads, run attention independently in each, concatenate, project. Each head can specialize (e.g. one for syntax, one for coreference). More expressive than a single attention with the same total dim.",
      "moduleId": "transformer-fundamentals",
      "tags": [
        "transformers",
        "attention",
        "multi-head"
      ]
    },
    {
      "id": "mutual-information",
      "term": "Mutual information",
      "aka": null,
      "definition": "How much learning one variable tells you about another, I(X, Y) = H(X) - H(X|Y), equivalently the KL divergence between the joint and the product of the marginals. Decision-tree information gain is exactly this quantity. Landmine: it is symmetric, despite the causal language people use around it.",
      "moduleId": "information-theory-core",
      "tags": [
        "information-theory",
        "mutual-information",
        "decision-trees"
      ]
    },
    {
      "id": "naive-bayes",
      "term": "Naive Bayes",
      "aka": null,
      "definition": "A generative classifier: model p(y)p(x|y) and invert with Bayes, assuming the features are independent given the class so one d-dimensional density becomes d one-dimensional ones. Fits in a single pass with no optimiser. Landmine: correlated features carry overlapping evidence and multiplying their likelihoods counts it once per feature, so the classifier is systematically overconfident and predict_proba is badly calibrated even where predict is fine.",
      "moduleId": "probability-foundations",
      "tags": [
        "classification",
        "generative",
        "bayes"
      ]
    },
    {
      "id": "neuron",
      "term": "Neuron",
      "aka": null,
      "definition": "A neuron takes inputs $a_1, \\dots, a_n$ and produces one number, $a = \\sigma(w_1 a_1 + w_2 a_2 + \\dots + w_n a_n + b)$. Each input has a weight $w_i$, a bias $b$ is added to the weighted sum, and the sum goes through an activation function $\\sigma$.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "neuron"
      ]
    },
    {
      "id": "nms",
      "term": "NMS",
      "aka": "Non-Maximum Suppression",
      "definition": "A greedy post-processing step for object detection: sort predictions by score, greedily keep the top, discard any that overlap it (IoU > threshold), repeat. Removes duplicate boxes for the same object.",
      "moduleId": "object-detection-fpn-yolo",
      "tags": [
        "object-detection",
        "post-processing",
        "iou"
      ]
    },
    {
      "id": "normal-equations",
      "term": "Normal equations",
      "aka": null,
      "definition": "AᵀAᴅ = Aᵀb: the closed-form solution to least squares. np.linalg.solve works. Lstsq or QR is better conditioned.",
      "moduleId": "linear-algebra-i-vectors-matrices-systems-transformations",
      "tags": [
        "linear-algebra",
        "regression",
        "ols"
      ]
    },
    {
      "id": "one-hot-encoding",
      "term": "One-hot encoding",
      "aka": null,
      "definition": "Represent a discrete class as a vector with a 1 in the class’s position, 0 elsewhere. What \"labels\" look like when computing cross-entropy against a softmax output.",
      "moduleId": "data-processing-features",
      "tags": [
        "encoding",
        "classification"
      ]
    },
    {
      "id": "overfitting",
      "term": "Overfitting",
      "aka": null,
      "definition": "Training loss keeps dropping while validation loss rises: the model is memorizing rather than generalizing. Counter with regularization, more data, less capacity, or early stopping.",
      "moduleId": "bias-variance-cross-validation",
      "tags": [
        "generalization",
        "overfitting",
        "regularization"
      ]
    },
    {
      "id": "parameters",
      "term": "Parameters",
      "aka": null,
      "definition": "The numbers a network learns: every weight and every bias. A 784-16-16-10 network has 13,002 of them, and training means finding good values for all of them.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "training"
      ]
    },
    {
      "id": "pca",
      "term": "PCA",
      "aka": "Principal Component Analysis",
      "definition": "Project data onto the top-k eigenvectors of the covariance matrix: the directions of maximum variance. Best linear low-rank approximation. Always center (and often standardize) first.",
      "moduleId": "principal-component-analysis",
      "tags": [
        "pca",
        "dimensionality-reduction",
        "eigendecomposition"
      ]
    },
    {
      "id": "perplexity",
      "term": "Perplexity",
      "aka": null,
      "definition": "Two raised to an entropy, which turns a distribution into an effective count. In t-SNE it is the target number of neighbours each point attends to, set instead of the per-point Gaussian bandwidth. Landmine: the same word means effective vocabulary size for a language model, so always check which sense is meant.",
      "moduleId": "nonlinear-dimensionality-reduction",
      "tags": [
        "t-sne",
        "perplexity",
        "entropy",
        "dimensionality-reduction"
      ]
    },
    {
      "id": "policy-gradient",
      "term": "Policy gradient",
      "aka": null,
      "definition": "Optimising a policy directly by the identity grad E[R] = E[grad log pi times R], which turns an integral you cannot compute into an expectation you can sample. Every dynamics term drops out when you expand it, which is what makes the method model-free. Landmine: it is on-policy, so data is stale the moment you update.",
      "moduleId": null,
      "tags": [
        "reinforcement-learning",
        "policy-gradient",
        "reinforce",
        "ppo"
      ]
    },
    {
      "id": "positional-encoding",
      "term": "Positional encoding",
      "aka": null,
      "definition": "Attention compares tokens by their contents only, and a weighted sum is the same in any order, so Transformers add per-position vectors to embed order. Options: fixed sinusoids (original), learned embeddings (BERT), RoPE (modern LLMs).",
      "moduleId": "relative-position-rope-positional-methods",
      "tags": [
        "transformers",
        "positional-encoding"
      ]
    },
    {
      "id": "positive-semidefinite",
      "term": "Positive semidefinite",
      "aka": "PSD",
      "definition": "A symmetric matrix whose quadratic form is never negative. For a symmetric matrix this is equivalent to having only nonnegative eigenvalues. A covariance matrix is positive semidefinite because $x^\\top\\Sigma x$ is a variance.",
      "moduleId": "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
      "tags": [
        "linear-algebra",
        "matrix"
      ]
    },
    {
      "id": "pretraining",
      "term": "Pretraining",
      "aka": null,
      "definition": "Train a model on a large, generic dataset before fine-tuning on the downstream task. For LLMs: next-token prediction on trillions of tokens. Everything downstream is a rounding error in compute compared to this.",
      "moduleId": "sequence-models-transfer-learning",
      "tags": [
        "pretraining",
        "transfer-learning",
        "llm"
      ]
    },
    {
      "id": "pytorch",
      "term": "PyTorch",
      "aka": null,
      "definition": "The dominant deep-learning framework. Dynamic (\"define-by-run\") graphs, first-class Python, autograd, and a huge ecosystem. What every notebook in this guide is written in.",
      "moduleId": "pytorch-tensors-autograd",
      "tags": [
        "pytorch",
        "framework"
      ]
    },
    {
      "id": "q-learning",
      "term": "Q-learning",
      "aka": null,
      "definition": "Learns the optimal action-value function by bootstrapping off the max over next actions, regardless of the policy actually generating the data. Off-policy, so it learns the optimal policy while behaving exploratorily. Landmine: SARSA looks almost identical but bootstraps off the action actually taken, and converges somewhere different.",
      "moduleId": "reinforcement-learning",
      "tags": [
        "reinforcement-learning",
        "q-learning",
        "off-policy",
        "td-learning"
      ]
    },
    {
      "id": "qr-decomposition",
      "term": "QR decomposition",
      "aka": null,
      "definition": "Factor A = QR with Q orthonormal and R upper triangular. Solving least squares through R rather than through the normal equations avoids forming A^T A, which squares the condition number and so halves the digits you keep. This is the standard reason a library solver beats the textbook formula.",
      "moduleId": "linear-algebra-i-vectors-matrices-systems-transformations",
      "tags": [
        "linear-algebra",
        "least-squares",
        "numerics"
      ]
    },
    {
      "id": "r-squared",
      "term": "R-squared",
      "aka": "Coefficient of determination",
      "definition": "One minus the squared error divided by the squared error of predicting the mean of the data being scored. It is 1 for a perfect fit and 0 for that constant, and on test data it can be any negative number. Landmine: a constant fitted on training data scores below zero whenever the target drifts, by exactly the squared drift over the test variance.",
      "moduleId": "linear-logistic-regression",
      "tags": [
        "regression",
        "metrics",
        "evaluation"
      ]
    },
    {
      "id": "rayleigh-quotient",
      "term": "Rayleigh quotient",
      "aka": null,
      "definition": "For a symmetric A, the ratio x^T A x over x^T x. Its maximum over all x is the largest eigenvalue and its minimum the smallest, with the optimising x being the matching eigenvector. This is the bridge from direction of greatest variance to top eigenvector of the covariance matrix, so it is what makes PCA an eigenproblem.",
      "moduleId": "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
      "tags": [
        "linear-algebra",
        "eigenvalues",
        "pca"
      ]
    },
    {
      "id": "regularization",
      "term": "Regularization",
      "aka": null,
      "definition": "Any technique that shrinks the effective hypothesis space to fight overfitting. L2 (weight decay), L1 (sparsity), dropout, early stopping, data augmentation: all count.",
      "moduleId": "neural-network-regularization-dropout-batchnorm",
      "tags": [
        "regularization",
        "generalization"
      ]
    },
    {
      "id": "relu",
      "term": "ReLU",
      "aka": "Rectified Linear Unit",
      "definition": "max(0, x). The default deep-learning nonlinearity: cheap, no vanishing gradients for positive inputs, sparse activations. Variants (LeakyReLU, GELU, SiLU) tweak the negative side.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "activation",
        "relu",
        "nonlinearity"
      ]
    },
    {
      "id": "reparameterization-trick",
      "term": "Reparameterization trick",
      "aka": null,
      "definition": "Rewrite z ~ N(μ, σ²) as z = μ + σε with ε ~ N(0,1). Gradients now flow through μ and σ, which is what makes a VAE trainable end-to-end.",
      "moduleId": "vae-encoder-decoder-generation",
      "tags": [
        "vae",
        "variational-inference",
        "gradients"
      ]
    },
    {
      "id": "ridge-regression",
      "term": "Ridge regression",
      "aka": null,
      "definition": "Least squares with an L2 penalty: min ‖Aᴅ-b‖² + λ‖ᴅ‖². Closed form: (AᵀA + λI)⁻¹Aᵀb. Always invertible for λ > 0.",
      "moduleId": "linear-logistic-regression",
      "tags": [
        "regression",
        "regularization",
        "l2"
      ]
    },
    {
      "id": "rmse",
      "term": "RMSE",
      "aka": "Root mean squared error",
      "definition": "The square root of the mean squared error, in the units of the target. It ranks models exactly as MSE does and weights large errors heavily. Landmine: pooled over several targets it is dominated by whichever target has the largest scale.",
      "moduleId": "linear-logistic-regression",
      "tags": [
        "regression",
        "metrics",
        "evaluation"
      ]
    },
    {
      "id": "rmsnorm",
      "term": "RMSNorm",
      "aka": null,
      "definition": "LayerNorm with the mean subtraction removed, keeping only the rescaling by the root mean square. Cheaper and empirically as good, which is why most current large language models use it. Landmine: it is exactly LayerNorm when the input is already zero-mean, and differs precisely by giving up shift-invariance.",
      "moduleId": null,
      "tags": [
        "normalization",
        "rmsnorm",
        "layernorm",
        "transformers"
      ]
    },
    {
      "id": "roialign",
      "term": "RoIAlign",
      "aka": null,
      "definition": "Extracts a fixed-size grid from a region without rounding its coordinates onto the feature grid, sampling with bilinear interpolation instead. RoIPool rounds the region’s edges to whole feature cells, which can move it onto the wrong cells: a shift a class label hardly notices but a mask does. The reason Mask R-CNN works.",
      "moduleId": "object-detection-fpn-yolo",
      "tags": [
        "detection",
        "segmentation",
        "alignment"
      ]
    },
    {
      "id": "rope",
      "term": "RoPE",
      "aka": "Rotary Position Embedding",
      "definition": "Rotary position embedding: rotate q and k by an angle proportional to position, so that because rotations are orthogonal and additive the attention score depends only on the offset. Applied absolutely, consumed relatively, and cache-compatible because keys are rotated once. Landmine: it is applied to q and k only, never to v.",
      "moduleId": "relative-position-rope-positional-methods",
      "tags": [
        "positional-encoding",
        "rope",
        "attention",
        "long-context"
      ]
    },
    {
      "id": "seasonal-naive-forecast",
      "term": "Seasonal naive forecast",
      "aka": null,
      "definition": "Predict each future value with the value one season earlier, such as the same weekday last week. It needs no training, and it is the baseline a forecasting model must beat before it has learned anything.",
      "moduleId": null,
      "tags": [
        "time-series",
        "forecasting",
        "baselines"
      ]
    },
    {
      "id": "self-attention",
      "term": "Self-attention",
      "aka": null,
      "definition": "Attention where Q, K, and V all come from the same input sequence. Every token attends to every other token. That’s where a Transformer’s power comes from.",
      "moduleId": "transformer-fundamentals",
      "tags": [
        "attention",
        "self-attention",
        "transformers"
      ]
    },
    {
      "id": "sgd",
      "term": "SGD",
      "aka": "Stochastic Gradient Descent",
      "definition": "Gradient descent using a random subset (mini-batch) of the data each step. Introduces noise that helps escape saddles and (surprisingly) generalizes better than full-batch. All the \"real\" optimizers (Adam, RMSprop) are SGD variants.",
      "moduleId": "optimization-adam-sgd",
      "tags": [
        "optimization",
        "sgd",
        "stochastic"
      ]
    },
    {
      "id": "sigmoid",
      "term": "Sigmoid",
      "aka": null,
      "definition": "σ(x) = 1/(1+e⁻ˣ). Squashes reals to (0,1). Its derivative σ(1-σ) is at most 0.25: a vanishing-gradient problem in deep nets. Fine as an output for binary classification.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "activation",
        "sigmoid",
        "vanishing-gradients"
      ]
    },
    {
      "id": "softmax",
      "term": "Softmax",
      "aka": null,
      "definition": "exp(x_i) / Σ exp(x_j). Turns logits into a probability distribution. The multi-class generalization of sigmoid. Compute with the max-subtraction trick for numerical stability.",
      "moduleId": "transformer-fundamentals",
      "tags": [
        "softmax",
        "classification",
        "numerical-stability"
      ]
    },
    {
      "id": "spectral-radius",
      "term": "Spectral radius",
      "aka": null,
      "definition": "The largest absolute value among the eigenvalues of a square matrix. It sets how fast long products grow or shrink, since the k-th root of the norm of A^k tends to it (Gelfand). Landmine: for a matrix that is not symmetric it can sit far below the largest singular value, so a bound in the largest singular value can be off by many orders of magnitude over many steps.",
      "moduleId": "sequence-models-transfer-learning",
      "tags": [
        "linear-algebra",
        "eigenvalues",
        "rnn"
      ]
    },
    {
      "id": "spectral-theorem",
      "term": "Spectral theorem",
      "aka": null,
      "definition": "A real symmetric matrix has an orthonormal basis of real eigenvectors. In these coordinates, applying $A$ simply scales each coordinate by an eigenvalue. This makes quadratic forms, Hessian curvature, and covariance matrices easier to interpret.",
      "moduleId": "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
      "tags": [
        "linear-algebra",
        "eigenvalues"
      ]
    },
    {
      "id": "support-vector",
      "term": "Support vector",
      "aka": null,
      "definition": "A training point on or inside the margin: the only points that determine the SVM’s decision boundary. Deleting everything else leaves the same classifier.",
      "moduleId": "kernel-methods-svms",
      "tags": [
        "svm",
        "geometry"
      ]
    },
    {
      "id": "svd",
      "term": "SVD",
      "aka": "Singular Value Decomposition",
      "definition": "A = UΣVᵀ with orthogonal U, V and non-negative diagonal Σ. Every matrix has one. Truncating to the top-k singular values gives the best rank-k approximation (Eckart-Young).",
      "moduleId": "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
      "tags": [
        "linear-algebra",
        "svd",
        "dimensionality-reduction"
      ]
    },
    {
      "id": "t-sne",
      "term": "t-SNE",
      "aka": null,
      "definition": "A neighbour-preserving embedding that matches a Gaussian neighbour distribution in high dimensions to a heavy-tailed Student-t one in two, minimising forward KL. Landmine: because forward KL barely penalises distant pairs, cluster sizes and between-cluster distances in the plot carry no information.",
      "moduleId": "nonlinear-dimensionality-reduction",
      "tags": [
        "t-sne",
        "dimensionality-reduction",
        "visualization",
        "kl-divergence"
      ]
    },
    {
      "id": "tanh",
      "term": "Tanh",
      "aka": "hyperbolic tangent",
      "definition": "$\\tanh(z) = (e^{z} - e^{-z})/(e^{z} + e^{-z})$, a sigmoid rescaled to run from −1 to 1.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "activation",
        "nonlinearity",
        "tanh"
      ]
    },
    {
      "id": "taylor-expansion",
      "term": "Taylor expansion",
      "aka": null,
      "definition": "Approximate a smooth function near a point using its derivatives there. For a twice continuously differentiable function, the linear term captures slope and the quadratic term captures curvature. The approximation is local: a useful small-step estimate need not remain accurate far from $x$. An infinite Taylor series requires stronger conditions to equal the function.",
      "moduleId": "calculus-gradients",
      "tags": [
        "calculus",
        "approximation"
      ]
    },
    {
      "id": "teacher-forcing",
      "term": "Teacher forcing",
      "aka": null,
      "definition": "Training a sequence model on the true prefix at every position rather than on its own predictions, which together with a causal mask gives all predictions in one pass. In code it is a one-token shift, inputs tokens[:-1] and targets tokens[1:]. Landmine: forget the shift and the model learns to copy its input, with a training loss that looks superb.",
      "moduleId": "sequence-models-transfer-learning",
      "tags": [
        "language-models",
        "training",
        "teacher-forcing"
      ]
    },
    {
      "id": "temperature",
      "term": "Temperature",
      "aka": null,
      "definition": "Divide logits by T > 0 before softmax. T < 1 sharpens (peaky), T > 1 flattens (uniform). Used for calibration, sampling, and controlling exploration in RL / LLM inference.",
      "moduleId": "gpt-causal-lm",
      "tags": [
        "softmax",
        "temperature",
        "calibration"
      ]
    },
    {
      "id": "threshold-unit",
      "term": "Threshold unit",
      "aka": null,
      "definition": "A unit that outputs 1 when its weighted sum is at least 0, and 0 otherwise. A layer of them can carve any convex polygon out of the plane: one hidden unit per side, each firing on that side's half-plane, and an output unit that fires only when all of them do. Its derivative is 0 everywhere it is defined, so a network of threshold units cannot be trained by gradient descent; the weights have to be set by hand.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "hand-built"
      ]
    },
    {
      "id": "timestep-embedding",
      "term": "Timestep embedding",
      "aka": null,
      "definition": "How a diffusion denoiser is told the noise level: map t to sinusoids, pass them through a small MLP, and add the result per channel in every residual block, or use it to scale and shift the features. It is needed because the best noise prediction for the same input differs from one noise level to another.",
      "moduleId": "generative-adding-noise",
      "tags": [
        "diffusion",
        "conditioning",
        "embeddings"
      ]
    },
    {
      "id": "top-p-sampling",
      "term": "Top-p sampling",
      "aka": "Nucleus sampling",
      "definition": "Sample only from the smallest set of most probable tokens whose total probability reaches p, renormalized. Unlike top-k it adapts: many tokens survive when the model is unsure, few when it is confident.",
      "moduleId": "gpt-causal-lm",
      "tags": [
        "decoding",
        "sampling",
        "language-models"
      ]
    },
    {
      "id": "transformer",
      "term": "Transformer",
      "aka": null,
      "definition": "An architecture built from self-attention + MLP blocks with residual connections and LayerNorm. Powers almost every modern large model, from GPT to Stable Diffusion to AlphaFold.",
      "moduleId": "transformer-fundamentals",
      "tags": [
        "transformers",
        "attention"
      ]
    },
    {
      "id": "umap",
      "term": "UMAP",
      "aka": null,
      "definition": "A neighbour-preserving embedding like t-SNE but faster, able to transform unseen points, and retaining somewhat more global structure. Its knobs are n_neighbors and min_dist. Landmine: more global structure than t-SNE still does not mean between-cluster distances are measurable.",
      "moduleId": "nonlinear-dimensionality-reduction",
      "tags": [
        "umap",
        "dimensionality-reduction",
        "visualization"
      ]
    },
    {
      "id": "unet",
      "term": "UNet",
      "aka": null,
      "definition": "An encoder-decoder with skip connections between corresponding levels. Originally for biomedical segmentation. Now the workhorse denoiser inside diffusion models.",
      "moduleId": "semantic-segmentation-unet-masks",
      "tags": [
        "segmentation",
        "unet",
        "diffusion"
      ]
    },
    {
      "id": "universal-approximation-theorem",
      "term": "Universal approximation theorem",
      "aka": null,
      "definition": "A single hidden layer, made wide enough, can approximate any continuous function as closely as you like. It needs inputs in a closed, bounded range such as $[0, 1]$, and an activation that is continuous and not a polynomial. The theorem says nothing about how wide, or whether training will find the weights.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "approximation"
      ]
    },
    {
      "id": "vae",
      "term": "VAE",
      "aka": "Variational Autoencoder",
      "definition": "A probabilistic autoencoder: the encoder outputs a distribution over z, the decoder maps z back to x. Trained by maximizing the ELBO.",
      "moduleId": "vae-encoder-decoder-generation",
      "tags": [
        "vae",
        "generative-models",
        "variational-inference"
      ]
    },
    {
      "id": "variance",
      "term": "Variance",
      "aka": null,
      "definition": "The average squared distance from the mean. Squaring prevents deviations above and below the mean from cancelling. Variance is measured in squared units; standard deviation takes its square root to return to the original units.",
      "moduleId": "expectations-multivariate-distributions",
      "tags": [
        "probability",
        "statistics"
      ]
    },
    {
      "id": "vit",
      "term": "ViT",
      "aka": "Vision Transformer",
      "definition": "Split an image into non-overlapping patches, linearly embed each patch as a token, feed to a Transformer. At scale, matches or beats CNNs on ImageNet.",
      "moduleId": "vision-transformers-vit-attention",
      "tags": [
        "vit",
        "transformers",
        "vision"
      ]
    },
    {
      "id": "weight",
      "term": "Weight",
      "aka": null,
      "definition": "The number a neuron multiplies one of its inputs by. The weights say what the neuron responds to: a positive weight means that input pushes the output up, a negative weight pushes it down, and a weight of 0 means the input is ignored.",
      "moduleId": "neural-networks-perceptrons-mlps-activations",
      "tags": [
        "neural-networks",
        "mlp",
        "parameters"
      ]
    },
    {
      "id": "weight-decay",
      "term": "Weight decay",
      "aka": null,
      "definition": "L2 regularization implemented as a shrinkage step in the optimizer: w ← w - η∇L(w) - ηλw. Equivalent to L2 penalty for vanilla SGD, not equivalent for Adam (hence AdamW).",
      "moduleId": "advanced-optimization",
      "tags": [
        "regularization",
        "weight-decay",
        "adamw"
      ]
    },
    {
      "id": "word2vec",
      "term": "Word2Vec",
      "aka": null,
      "definition": "A pair of shallow neural models (skip-gram and CBOW) that learn word embeddings by predicting neighboring words. Landmark of \"vector space semantics\", king - man + woman ≈ queen.",
      "moduleId": "natural-language-processing-tokenization-embeddings",
      "tags": [
        "embeddings",
        "nlp",
        "word2vec"
      ]
    }
  ];
})();
