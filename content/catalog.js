(function () {
  const modules = [
    {
      "id": "python-basics",
      "title": "Python Basics",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Enough Python for every later module.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.23-PY7-python-basics.md",
      "code": "PY7",
      "number": "1.23",
      "frequency": 4,
      "prerequisiteTitles": []
    },
    {
      "id": "programming-essentials",
      "title": "NumPy Programming",
      "section": "round1",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Arrays, vectorized code, broadcasting and indexing.",
      "prerequisites": [
        "python-basics"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.1-PY1-programming-essentials.md",
      "code": "PY1",
      "number": "1.1",
      "frequency": 4,
      "prerequisiteTitles": [
        "Python Basics"
      ]
    },
    {
      "id": "pandas-data-cleaning-storing",
      "title": "Pandas Data Cleaning",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Loading, cleaning and reshaping a table.",
      "prerequisites": [
        "programming-essentials"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.2-PY2-pandas-data-cleaning-storing.md",
      "code": "PY2",
      "number": "1.2",
      "frequency": 3,
      "prerequisiteTitles": [
        "NumPy Programming"
      ]
    },
    {
      "id": "plotting-matplotlib-seaborn",
      "title": "Data Visualization",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Clear plots with Matplotlib.",
      "prerequisites": [
        "programming-essentials",
        "pandas-data-cleaning-storing"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.3-PY3-plotting-matplotlib-seaborn.md",
      "code": "PY3",
      "number": "1.3",
      "frequency": 3,
      "prerequisiteTitles": [
        "NumPy Programming",
        "Pandas Data Cleaning"
      ]
    },
    {
      "id": "colab-markdown-notebooks",
      "title": "Colab Notebooks",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Running a notebook and writing answers a grader can read.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.4-PY4-colab-markdown-notebooks.md",
      "code": "PY4",
      "number": "1.4",
      "frequency": 3,
      "prerequisiteTitles": []
    },
    {
      "id": "scikit-learn-api-ml-pipelines",
      "title": "Scikit-Learn Pipelines",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Fit, tune and score models without leaking test data.",
      "prerequisites": [
        "programming-essentials",
        "pandas-data-cleaning-storing"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.5-PY5-scikit-learn-api-ml-pipelines.md",
      "code": "PY5",
      "number": "1.5",
      "frequency": 3,
      "prerequisiteTitles": [
        "NumPy Programming",
        "Pandas Data Cleaning"
      ]
    },
    {
      "id": "why-pytorch-not-tensorflow",
      "title": "Why PyTorch, not TensorFlow",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "The framework rule, define-by-run, and how autograd computes a gradient.",
      "prerequisites": [
        "calculus-gradients",
        "programming-essentials"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.24-PY8-why-pytorch-not-tensorflow.md",
      "code": "PY8",
      "number": "1.24",
      "frequency": 1,
      "prerequisiteTitles": [
        "Calculus and Gradients",
        "NumPy Programming"
      ]
    },
    {
      "id": "pytorch-tensors-autograd",
      "title": "PyTorch Tensors & Autograd",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Tensor shapes, views, broadcasting, reductions, and autograd in code.",
      "prerequisites": [
        "why-pytorch-not-tensorflow",
        "programming-essentials"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.25-PY9-pytorch-tensors-autograd.md",
      "code": "PY9",
      "number": "1.25",
      "frequency": 4,
      "prerequisiteTitles": [
        "Why PyTorch, not TensorFlow",
        "NumPy Programming"
      ]
    },
    {
      "id": "training-loop-in-pytorch",
      "title": "Writing a Training Loop in PyTorch",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Batches from a DataLoader, the five-line training step, evaluation, and the loop that ties them together, on MNIST.",
      "prerequisites": [
        "pytorch-tensors-autograd",
        "loss-functions"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.26-PY10-training-loop-in-pytorch.md",
      "code": "PY10",
      "number": "1.26",
      "frequency": 4,
      "prerequisiteTitles": [
        "PyTorch Tensors & Autograd",
        "Loss Functions"
      ]
    },
    {
      "id": "linear-algebra-i-vectors-matrices-systems-transformations",
      "title": "Vectors and Matrices",
      "section": "round1",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Dot products, matrices, rank and least squares.",
      "prerequisites": [
        "programming-essentials"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.7-MA1-linear-algebra-i-vectors-matrices-systems-transformations.md",
      "code": "MA1",
      "number": "1.7",
      "frequency": 4,
      "prerequisiteTitles": [
        "NumPy Programming"
      ]
    },
    {
      "id": "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
      "title": "Eigenvalues and Spectral Theory",
      "section": "round1",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Eigenvalues, symmetric matrices and the SVD.",
      "prerequisites": [
        "linear-algebra-i-vectors-matrices-systems-transformations"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.8-MA2-linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory.md",
      "code": "MA2",
      "number": "1.8",
      "frequency": 4,
      "prerequisiteTitles": [
        "Vectors and Matrices"
      ]
    },
    {
      "id": "calculus-gradients",
      "title": "Calculus and Gradients",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Derivatives, the chain rule, gradients and Hessians.",
      "prerequisites": [
        "linear-algebra-i-vectors-matrices-systems-transformations",
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.9-MA3-calculus-gradients.md",
      "code": "MA3",
      "number": "1.9",
      "frequency": 4,
      "prerequisiteTitles": [
        "Vectors and Matrices",
        "Eigenvalues and Spectral Theory"
      ]
    },
    {
      "id": "probability-foundations",
      "title": "Probability and Bayes",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Counting, conditional probability and Bayes' theorem.",
      "prerequisites": [
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
        "programming-essentials"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.10-PS1-probability-foundations.md",
      "code": "PS1",
      "number": "1.10",
      "frequency": 2,
      "prerequisiteTitles": [
        "Eigenvalues and Spectral Theory",
        "NumPy Programming"
      ]
    },
    {
      "id": "expectations-multivariate-distributions",
      "title": "Expectation and Joint Distributions",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Expectation, variance, covariance and joint distributions.",
      "prerequisites": [
        "probability-foundations",
        "linear-algebra-i-vectors-matrices-systems-transformations"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.11-PS2-expectations-multivariate-distributions.md",
      "code": "PS2",
      "number": "1.11",
      "frequency": 3,
      "prerequisiteTitles": [
        "Probability and Bayes",
        "Vectors and Matrices"
      ]
    },
    {
      "id": "statistics-hypothesis-testing",
      "title": "Statistical Inference",
      "section": "round1",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Estimating a parameter and saying how sure you are.",
      "prerequisites": [
        "expectations-multivariate-distributions"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.12-PS3-statistics-hypothesis-testing.md",
      "code": "PS3",
      "number": "1.12",
      "frequency": 1,
      "prerequisiteTitles": [
        "Expectation and Joint Distributions"
      ]
    },
    {
      "id": "data-processing-features",
      "title": "Feature Engineering & Data Preparation",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Scaling, encoding, missing values and leakage.",
      "prerequisites": [
        "pandas-data-cleaning-storing",
        "decision-trees-knn"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.13-ML1-data-processing-features.md",
      "code": "ML1",
      "number": "1.13",
      "frequency": 3,
      "prerequisiteTitles": [
        "Pandas Data Cleaning",
        "Decision Trees & k-Nearest Neighbors"
      ]
    },
    {
      "id": "linear-logistic-regression",
      "title": "Linear & Logistic Regression",
      "section": "round1",
      "level": "Core",
      "time": "2 weeks",
      "summary": "Least squares and logistic regression, by hand and in NumPy.",
      "prerequisites": [
        "linear-algebra-i-vectors-matrices-systems-transformations",
        "calculus-gradients",
        "statistics-hypothesis-testing"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.27-ML8-linear-logistic-regression.md",
      "code": "ML8",
      "number": "1.27",
      "frequency": 4,
      "prerequisiteTitles": [
        "Vectors and Matrices",
        "Calculus and Gradients",
        "Statistical Inference"
      ]
    },
    {
      "id": "loss-functions",
      "title": "Loss Functions",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Squared error, cross-entropy and hinge, and where each comes from.",
      "prerequisites": [
        "statistics-hypothesis-testing",
        "linear-logistic-regression"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.28-ML9-loss-functions.md",
      "code": "ML9",
      "number": "1.28",
      "frequency": 3,
      "prerequisiteTitles": [
        "Statistical Inference",
        "Linear & Logistic Regression"
      ]
    },
    {
      "id": "kernel-methods-svms",
      "title": "Support Vector Machines",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "The maximum-margin classifier, the hinge loss and kernels.",
      "prerequisites": [
        "linear-logistic-regression"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.15-ML3-kernel-methods-svms.md",
      "code": "ML3",
      "number": "1.15",
      "frequency": 1,
      "prerequisiteTitles": [
        "Linear & Logistic Regression"
      ]
    },
    {
      "id": "decision-trees-knn",
      "title": "Decision Trees & k-Nearest Neighbors",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Nearest neighbors and decision trees, by hand and in scikit-learn.",
      "prerequisites": [
        "expectations-multivariate-distributions",
        "programming-essentials"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.29-ML10-decision-trees-knn.md",
      "code": "ML10",
      "number": "1.29",
      "frequency": 3,
      "prerequisiteTitles": [
        "Expectation and Joint Distributions",
        "NumPy Programming"
      ]
    },
    {
      "id": "tree-based-models-ensembles",
      "title": "Ensemble Learning",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Bagging, random forests and boosting.",
      "prerequisites": [
        "decision-trees-knn"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.16-ML4-tree-based-models-ensembles.md",
      "code": "ML4",
      "number": "1.16",
      "frequency": 1,
      "prerequisiteTitles": [
        "Decision Trees & k-Nearest Neighbors"
      ]
    },
    {
      "id": "clustering",
      "title": "k-Means Clustering",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Lloyd's algorithm, k-means++ and choosing k.",
      "prerequisites": [
        "linear-algebra-i-vectors-matrices-systems-transformations",
        "programming-essentials"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.17-ML5-clustering.md",
      "code": "ML5",
      "number": "1.17",
      "frequency": 2,
      "prerequisiteTitles": [
        "Vectors and Matrices",
        "NumPy Programming"
      ]
    },
    {
      "id": "clustering-methods",
      "title": "Clustering Methods",
      "section": "round1",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "For two elongated groups, distances to a centre, distances between nearest members, and density connectivity may suggest different partitions.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.31-ML5-clustering-methods.md",
      "code": "ML5",
      "number": "1.31",
      "prerequisiteTitles": []
    },
    {
      "id": "neural-networks-perceptrons-mlps-activations",
      "title": "The Multilayer Perceptron",
      "section": "round1",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Neurons, layers, activation functions, and the network that reads handwritten digits.",
      "prerequisites": [
        "linear-algebra-i-vectors-matrices-systems-transformations",
        "linear-logistic-regression",
        "pytorch-tensors-autograd"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.18-DL1-neural-networks-perceptrons-mlps-activations.md",
      "code": "DL1",
      "number": "1.18",
      "frequency": 4,
      "prerequisiteTitles": [
        "Vectors and Matrices",
        "Linear & Logistic Regression",
        "PyTorch Tensors & Autograd"
      ]
    },
    {
      "id": "neural-network-regularization-dropout-batchnorm",
      "title": "Core Layers (Affine, Normalization, Dropout)",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Batch normalization, layer normalization, and dropout, by hand and as your own nn.Module layers.",
      "prerequisites": [
        "neural-networks-perceptrons-mlps-activations",
        "expectations-multivariate-distributions"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.19-DL2-neural-network-regularization-dropout-batchnorm.md",
      "code": "DL2",
      "number": "1.19",
      "frequency": 3,
      "prerequisiteTitles": [
        "The Multilayer Perceptron",
        "Expectation and Joint Distributions"
      ]
    },
    {
      "id": "optimization-adam-sgd",
      "title": "Optimizers & Training Tricks",
      "section": "round1",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "What optimizer.step() does: momentum, Adam, weight decay, schedules, early stopping, and the initialization that keeps a deep network trainable.",
      "prerequisites": [
        "training-loop-in-pytorch",
        "advanced-optimization"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.21-DL4-optimization-adam-sgd.md",
      "code": "DL4",
      "number": "1.21",
      "frequency": 3,
      "prerequisiteTitles": [
        "Writing a Training Loop in PyTorch",
        "Convex Optimization"
      ]
    },
    {
      "id": "convolutional-neural-networks",
      "title": "Convolutional Neural Networks",
      "section": "round1",
      "level": "Core",
      "time": "1 week",
      "summary": "Convolution, output size, parameter counting, pooling and receptive field, a small CNN in PyTorch, residual blocks and frozen backbones.",
      "prerequisites": [
        "neural-networks-perceptrons-mlps-activations",
        "neural-network-regularization-dropout-batchnorm",
        "programming-essentials",
        "training-loop-in-pytorch"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/1_Round_1/1.30-CV7-convolutional-neural-networks.md",
      "code": "CV7",
      "number": "1.30",
      "frequency": 3,
      "prerequisiteTitles": [
        "The Multilayer Perceptron",
        "Core Layers (Affine, Normalization, Dropout)",
        "NumPy Programming",
        "Writing a Training Loop in PyTorch"
      ]
    },
    {
      "id": "forward-propagation",
      "title": "Forward Propagation",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "The forward pass by hand and as a general NumPy loop, with the cache that backpropagation needs.",
      "prerequisites": [
        "neural-networks-perceptrons-mlps-activations",
        "linear-algebra-i-vectors-matrices-systems-transformations"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.15-MA6-forward-propagation.md",
      "code": "MA6",
      "number": "2.15",
      "frequency": 3,
      "prerequisiteTitles": [
        "The Multilayer Perceptron",
        "Vectors and Matrices"
      ]
    },
    {
      "id": "backpropagation-mechanics",
      "title": "Backpropagation by Hand",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "The chain rule organized as one backward sweep: the error signal, the outer-product gradients, and the step back through a layer, worked on a small network.",
      "prerequisites": [
        "forward-propagation",
        "calculus-gradients"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.1-MA4-backpropagation-mechanics.md",
      "code": "MA4",
      "number": "2.1",
      "frequency": 2,
      "prerequisiteTitles": [
        "Forward Propagation",
        "Calculus and Gradients"
      ]
    },
    {
      "id": "model-evaluation",
      "title": "Classification Metrics",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "The confusion matrix, precision, recall, F1 and AUC.",
      "prerequisites": [
        "linear-logistic-regression",
        "bias-variance-cross-validation"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.3-ML6-model-evaluation.md",
      "code": "ML6",
      "number": "2.3",
      "frequency": 3,
      "prerequisiteTitles": [
        "Linear & Logistic Regression",
        "Bias-Variance Tradeoff & Cross-Validation"
      ]
    },
    {
      "id": "bias-variance-cross-validation",
      "title": "Bias-Variance Tradeoff & Cross-Validation",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Where test error comes from and how to estimate it.",
      "prerequisites": [
        "statistics-hypothesis-testing",
        "linear-logistic-regression"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.16-ML11-bias-variance-cross-validation.md",
      "code": "ML11",
      "number": "2.16",
      "frequency": 3,
      "prerequisiteTitles": [
        "Statistical Inference",
        "Linear & Logistic Regression"
      ]
    },
    {
      "id": "principal-component-analysis",
      "title": "Principal Component Analysis",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "PCA from the covariance matrix and the SVD.",
      "prerequisites": [
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.17-ML12-principal-component-analysis.md",
      "code": "ML12",
      "number": "2.17",
      "frequency": 3,
      "prerequisiteTitles": [
        "Eigenvalues and Spectral Theory"
      ]
    },
    {
      "id": "nonlinear-dimensionality-reduction",
      "title": "Nonlinear Dimensionality Reduction: t-SNE and UMAP",
      "section": "round2",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "T-SNE and UMAP, and how to read their plots.",
      "prerequisites": [
        "principal-component-analysis"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.18-ML13-nonlinear-dimensionality-reduction.md",
      "code": "ML13",
      "number": "2.18",
      "frequency": 1,
      "prerequisiteTitles": [
        "Principal Component Analysis"
      ]
    },
    {
      "id": "semi-supervised-learning",
      "title": "Semi-Supervised Learning",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Pseudo-labeling and label propagation with few labels.",
      "prerequisites": [
        "decision-trees-knn",
        "kernel-methods-svms",
        "clustering",
        "clustering-methods"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.19-ML14-semi-supervised-learning.md",
      "code": "ML14",
      "number": "2.19",
      "frequency": 3,
      "prerequisiteTitles": [
        "Decision Trees & k-Nearest Neighbors",
        "Support Vector Machines",
        "k-Means Clustering",
        "Clustering Methods"
      ]
    },
    {
      "id": "time-series",
      "title": "Time Series",
      "section": "round2",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Windows, time-ordered evaluation and forecasting.",
      "prerequisites": [
        "bias-variance-cross-validation",
        "data-processing-features",
        "linear-logistic-regression"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.20-ML15-time-series.md",
      "code": "ML15",
      "number": "2.20",
      "frequency": 2,
      "prerequisiteTitles": [
        "Bias-Variance Tradeoff & Cross-Validation",
        "Feature Engineering & Data Preparation",
        "Linear & Logistic Regression"
      ]
    },
    {
      "id": "advanced-optimization",
      "title": "Convex Optimization",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Gradient descent, convexity and Lagrangian duality.",
      "prerequisites": [
        "calculus-gradients",
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.5-DL5-advanced-optimization.md",
      "code": "DL5",
      "number": "2.5",
      "frequency": 2,
      "prerequisiteTitles": [
        "Calculus and Gradients",
        "Eigenvalues and Spectral Theory"
      ]
    },
    {
      "id": "advanced-optimization-methods",
      "title": "Advanced Optimization",
      "section": "round2",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Simple objectives let us isolate why an optimizer makes progress or stalls.",
      "prerequisites": [
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
        "advanced-optimization"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.6-DL5-advanced-optimization-methods.md",
      "code": "DL5",
      "number": "2.6",
      "prerequisiteTitles": [
        "Eigenvalues and Spectral Theory",
        "Convex Optimization"
      ]
    },
    {
      "id": "vision-setup",
      "title": "1. Setup and pixels",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "The image classification task, and how an image becomes numbers.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.21-CV8-vision-setup.md",
      "code": "CV8",
      "number": "2.21",
      "frequency": 3,
      "prerequisiteTitles": []
    },
    {
      "id": "vision-convolution",
      "title": "2. Convolution and pooling",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Convolution filters, ReLU and pooling.",
      "prerequisites": [
        "neural-networks-perceptrons-mlps-activations",
        "optimization-adam-sgd",
        "programming-essentials",
        "pytorch-tensors-autograd"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.23-CV10-vision-convolution.md",
      "code": "CV10",
      "number": "2.23",
      "frequency": 3,
      "prerequisiteTitles": [
        "The Multilayer Perceptron",
        "Optimizers & Training Tricks",
        "NumPy Programming",
        "PyTorch Tensors & Autograd"
      ]
    },
    {
      "id": "vision-classify",
      "title": "3. Classifying, going deeper and transfer learning",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "A classifier head, deeper networks with residual connections, and reusing a trained network.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.25-CV12-vision-classify.md",
      "code": "CV12",
      "number": "2.25",
      "frequency": 3,
      "prerequisiteTitles": []
    },
    {
      "id": "hierarchical-vision-models",
      "title": "Hierarchical vision models: Swin, ConvNeXt and compound scaling",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "A useful backbone must move information across space without making high-resolution computation unaffordable.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.28-CV15-hierarchical-vision-models.md",
      "code": "CV15",
      "number": "2.28",
      "prerequisiteTitles": []
    },
    {
      "id": "semantic-segmentation-unet-masks",
      "title": "4. Segmentation and adversarial attacks",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "A label for every pixel, and inputs built to fool a classifier.",
      "prerequisites": [
        "convolutional-neural-networks"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.7-CV3-semantic-segmentation-unet-masks.md",
      "code": "CV3",
      "number": "2.7",
      "frequency": 1,
      "prerequisiteTitles": [
        "Convolutional Neural Networks"
      ]
    },
    {
      "id": "modern-vision-architectures-resnet-efficientnet-transfer",
      "title": "ResNets and Transfer Learning",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Understand how to adapt a pretrained backbone: what to freeze, what to train, at what rate, and when to doubt the transfer.",
      "prerequisites": [
        "convolutional-neural-networks",
        "semantic-segmentation-unet-masks",
        "neural-network-regularization-dropout-batchnorm",
        "bias-variance-cross-validation",
        "linear-logistic-regression"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.8-CV4-modern-vision-architectures-resnet-efficientnet-transfer.md",
      "code": "CV4",
      "number": "2.8",
      "prerequisiteTitles": [
        "Convolutional Neural Networks",
        "4. Segmentation and adversarial attacks",
        "Core Layers (Affine, Normalization, Dropout)",
        "Bias-Variance Tradeoff & Cross-Validation",
        "Linear & Logistic Regression"
      ]
    },
    {
      "id": "transformer-setup",
      "title": "1. Setup and input",
      "section": "round2",
      "level": "Core",
      "time": "2 weeks",
      "summary": "The next-token task, then text to token IDs, vectors and positions.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.30-TF10-transformer-setup.md",
      "code": "TF10",
      "number": "2.30",
      "frequency": 4,
      "prerequisiteTitles": []
    },
    {
      "id": "transformer-qkv",
      "title": "2. Attention scores",
      "section": "round2",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Queries, keys and values, a score for every pair of tokens, and the causal mask and softmax.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.34-TF14-transformer-qkv.md",
      "code": "TF14",
      "number": "2.34",
      "frequency": 4,
      "prerequisiteTitles": []
    },
    {
      "id": "transformer-weighted-sum",
      "title": "3. Attention output",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "The weighted sum of values, and several heads side by side.",
      "prerequisites": [
        "transformer-fundamentals"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.37-TF17-transformer-weighted-sum.md",
      "code": "TF17",
      "number": "2.37",
      "frequency": 4,
      "prerequisiteTitles": [
        "Self-Attention Fundamentals"
      ]
    },
    {
      "id": "transformer-residual-and-layernorm",
      "title": "4. The Transformer block",
      "section": "round2",
      "level": "Core",
      "time": "1-2 weeks",
      "summary": "Residual connections, LayerNorm, the feed-forward network, and stacking blocks.",
      "prerequisites": [
        "transformer-fundamentals",
        "neural-networks-perceptrons-mlps-activations"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.39-TF19-transformer-residual-and-layernorm.md",
      "code": "TF19",
      "number": "2.39",
      "frequency": 3,
      "prerequisiteTitles": [
        "Self-Attention Fundamentals",
        "The Multilayer Perceptron"
      ]
    },
    {
      "id": "transformer-predict",
      "title": "5. Predicting and generating",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Next-token probabilities, the training loss, and generating text.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.42-TF22-transformer-predict.md",
      "code": "TF22",
      "number": "2.42",
      "frequency": 2,
      "prerequisiteTitles": []
    },
    {
      "id": "transformer-fundamentals",
      "title": "Self-Attention Fundamentals",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Attention and a full transformer block implemented from scratch in NumPy, with the per-layer cost in FLOPs and memory.",
      "prerequisites": [
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
        "backpropagation-mechanics",
        "probability-foundations",
        "neural-networks-perceptrons-mlps-activations",
        "neural-network-regularization-dropout-batchnorm",
        "convolutional-neural-networks"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.9-TF1-transformer-fundamentals.md",
      "code": "TF1",
      "number": "2.9",
      "prerequisiteTitles": [
        "Eigenvalues and Spectral Theory",
        "Backpropagation by Hand",
        "Probability and Bayes",
        "The Multilayer Perceptron",
        "Core Layers (Affine, Normalization, Dropout)",
        "Convolutional Neural Networks"
      ]
    },
    {
      "id": "transformer-architectures-bert-mlm",
      "title": "BERT and Masked Language Modeling",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "How BERT learns by filling in blanks.",
      "prerequisites": [
        "transformer-residual-and-layernorm"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.10-TF2-transformer-architectures-bert-mlm.md",
      "code": "TF2",
      "number": "2.10",
      "frequency": 2,
      "prerequisiteTitles": [
        "4. The Transformer block"
      ]
    },
    {
      "id": "gpt-causal-lm",
      "title": "GPT and Causal Language Modeling",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Next-token prediction, the causal mask and generation.",
      "prerequisites": [
        "transformer-residual-and-layernorm"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.44-TF24-gpt-causal-lm.md",
      "code": "TF24",
      "number": "2.44",
      "frequency": 1,
      "prerequisiteTitles": [
        "4. The Transformer block"
      ]
    },
    {
      "id": "transformer-architectures-t5-encoder-decoder",
      "title": "T5 and Sequence-to-Sequence Models",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "Encoder-decoder Transformers and T5's text-in, text-out idea.",
      "prerequisites": [
        "transformer-architectures-bert-mlm",
        "gpt-causal-lm"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.12-TF4-transformer-architectures-t5-encoder-decoder.md",
      "code": "TF4",
      "number": "2.12",
      "frequency": 1,
      "prerequisiteTitles": [
        "BERT and Masked Language Modeling",
        "GPT and Causal Language Modeling"
      ]
    },
    {
      "id": "contrastive-learning",
      "title": "Contrastive Learning and CLIP",
      "section": "round2",
      "level": "Core",
      "time": "1 week",
      "summary": "InfoNCE, temperature, and how CLIP puts images and text in one space.",
      "prerequisites": [
        "linear-logistic-regression",
        "convolutional-neural-networks",
        "clustering",
        "neural-network-regularization-dropout-batchnorm",
        "modern-vision-architectures-resnet-efficientnet-transfer",
        "transformer-architectures-bert-mlm"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/2_Round_2/2.14-RP2-contrastive-learning.md",
      "code": "RP2",
      "number": "2.14",
      "frequency": 3,
      "prerequisiteTitles": [
        "Linear & Logistic Regression",
        "Convolutional Neural Networks",
        "k-Means Clustering",
        "Core Layers (Affine, Normalization, Dropout)",
        "ResNets and Transfer Learning",
        "BERT and Masked Language Modeling"
      ]
    },
    {
      "id": "information-theory-core",
      "title": "Information Theory for AI",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Entropy, cross-entropy, KL divergence, and mutual information.",
      "prerequisites": [
        "statistics-hypothesis-testing"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.1-PS4-information-theory-core.md",
      "code": "PS4",
      "number": "3.1",
      "frequency": 2,
      "prerequisiteTitles": [
        "Statistical Inference"
      ]
    },
    {
      "id": "vision-transformers-vit-attention",
      "title": "5. Vision Transformer and object detection",
      "section": "camp",
      "level": "Advanced",
      "time": "1-2 weeks",
      "summary": "Attention over image patches, and finding objects with boxes.",
      "prerequisites": [
        "modern-vision-architectures-resnet-efficientnet-transfer",
        "transformer-fundamentals",
        "convolutional-neural-networks",
        "model-evaluation",
        "neural-network-regularization-dropout-batchnorm"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.3-CV6-vision-transformers-vit-attention.md",
      "code": "CV6",
      "number": "3.3",
      "frequency": 2,
      "prerequisiteTitles": [
        "ResNets and Transfer Learning",
        "Self-Attention Fundamentals",
        "Convolutional Neural Networks",
        "Classification Metrics",
        "Core Layers (Affine, Normalization, Dropout)"
      ]
    },
    {
      "id": "natural-language-processing-tokenization-embeddings",
      "title": "Tokenization & Word Embeddings",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "WordPiece, Unigram, and byte-level BPE, word2vec skip-gram and CBOW, negative sampling, and GloVe.",
      "prerequisites": [
        "transformer-setup"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.4-NL1-natural-language-processing-tokenization-embeddings.md",
      "code": "NL1",
      "number": "3.4",
      "frequency": 2,
      "prerequisiteTitles": [
        "1. Setup and input"
      ]
    },
    {
      "id": "pre-training-fine-tuning",
      "title": "Pre-training & Fine-tuning",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Pretraining objectives, fine-tuning, and transfer.",
      "prerequisites": [
        "transformer-residual-and-layernorm",
        "training-loop-in-pytorch"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.14-NL3-pre-training-fine-tuning.md",
      "code": "NL3",
      "number": "3.14",
      "frequency": 2,
      "prerequisiteTitles": [
        "4. The Transformer block",
        "Writing a Training Loop in PyTorch"
      ]
    },
    {
      "id": "sequence-models-transfer-learning",
      "title": "Recurrent Networks: RNN, LSTM, GRU",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "One hidden state carried through a sequence, why its gradients vanish, how gates fix it, and the shapes nn.RNN, nn.LSTM and nn.GRU expect.",
      "prerequisites": [
        "neural-networks-perceptrons-mlps-activations",
        "backpropagation-mechanics"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.5-NL2-sequence-models-transfer-learning.md",
      "code": "NL2",
      "number": "3.5",
      "frequency": 0,
      "prerequisiteTitles": [
        "The Multilayer Perceptron",
        "Backpropagation by Hand"
      ]
    },
    {
      "id": "rag-retrieval-embeddings",
      "title": "Retrieval-Augmented Generation",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Letting a language model look things up.",
      "prerequisites": [
        "gpt-causal-lm",
        "natural-language-processing-tokenization-embeddings"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.6-TF5-rag-retrieval-embeddings.md",
      "code": "TF5",
      "number": "3.6",
      "frequency": 0,
      "prerequisiteTitles": [
        "GPT and Causal Language Modeling",
        "Tokenization & Word Embeddings"
      ]
    },
    {
      "id": "relative-position-rope-positional-methods",
      "title": "Positional Encoding and RoPE",
      "section": "camp",
      "level": "Advanced",
      "time": "1-2 weeks",
      "summary": "Relative encodings, rotary position embedding, relative position in linear attention, and stretching the context.",
      "prerequisites": [
        "transformer-setup",
        "transformer-qkv"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.7-TF6-relative-position-rope-positional-methods.md",
      "code": "TF6",
      "number": "3.7",
      "frequency": 3,
      "prerequisiteTitles": [
        "1. Setup and input",
        "2. Attention scores"
      ]
    },
    {
      "id": "cross-attention-multimodal-attention",
      "title": "Vision-Language Models",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "How models like LLaVA feed images to a language model.",
      "prerequisites": [
        "vision-transformers-vit-attention",
        "contrastive-learning",
        "gpt-causal-lm"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.8-TF7-cross-attention-multimodal-attention.md",
      "code": "TF7",
      "number": "3.8",
      "frequency": 0,
      "prerequisiteTitles": [
        "5. Vision Transformer and object detection",
        "Contrastive Learning and CLIP",
        "GPT and Causal Language Modeling"
      ]
    },
    {
      "id": "representation-learning-autoencoders-vaes-latent-spaces",
      "title": "2. Autoencoders and VAEs",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Compressing data through a bottleneck, and turning that into a generative model.",
      "prerequisites": [
        "clustering",
        "principal-component-analysis",
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
        "neural-networks-perceptrons-mlps-activations",
        "information-theory-core",
        "convolutional-neural-networks",
        "gpt-causal-lm",
        "probability-foundations",
        "model-evaluation"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/2.13-RP1-representation-learning-autoencoders-vaes-latent-spaces.md",
      "code": "RP1",
      "number": "2.13",
      "frequency": 2,
      "prerequisiteTitles": [
        "k-Means Clustering",
        "Principal Component Analysis",
        "Eigenvalues and Spectral Theory",
        "The Multilayer Perceptron",
        "Information Theory for AI",
        "Convolutional Neural Networks",
        "GPT and Causal Language Modeling",
        "Probability and Bayes",
        "Classification Metrics"
      ]
    },
    {
      "id": "autoregressive-generative-models",
      "title": "Autoregressive Generation",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Factor sequence or image distributions into conditional predictions and sample from them.",
      "prerequisites": [
        "natural-language-processing-tokenization-embeddings",
        "probability-foundations",
        "information-theory-core",
        "representation-learning-autoencoders-vaes-latent-spaces",
        "gpt-causal-lm",
        "transformer-predict",
        "contrastive-learning",
        "model-evaluation"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.9-RP4-autoregressive-generative-models.md",
      "code": "RP4",
      "number": "3.9",
      "prerequisiteTitles": [
        "Tokenization & Word Embeddings",
        "Probability and Bayes",
        "Information Theory for AI",
        "2. Autoencoders and VAEs",
        "GPT and Causal Language Modeling",
        "5. Predicting and generating",
        "Contrastive Learning and CLIP",
        "Classification Metrics"
      ]
    },
    {
      "id": "generative-setup",
      "title": "1. Setup",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "The data distribution, three approaches and the roadmap.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.15-GM3-generative-setup.md",
      "code": "GM3",
      "number": "3.15",
      "frequency": 3,
      "prerequisiteTitles": []
    },
    {
      "id": "generative-adversarial-networks",
      "title": "3. GAN",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Generator and discriminator, the best discriminator and mode collapse.",
      "prerequisites": [
        "representation-learning-autoencoders-vaes-latent-spaces",
        "gpt-causal-lm",
        "information-theory-core",
        "advanced-optimization",
        "probability-foundations",
        "model-evaluation",
        "linear-logistic-regression"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.11-GM1-generative-adversarial-networks.md",
      "code": "GM1",
      "number": "3.11",
      "frequency": 1,
      "prerequisiteTitles": [
        "2. Autoencoders and VAEs",
        "GPT and Causal Language Modeling",
        "Information Theory for AI",
        "Convex Optimization",
        "Probability and Bayes",
        "Classification Metrics",
        "Linear & Logistic Regression"
      ]
    },
    {
      "id": "generative-adding-noise",
      "title": "4. Diffusion and guidance",
      "section": "camp",
      "level": "Advanced",
      "time": "1-2 weeks",
      "summary": "Adding noise, learning to remove it, and steering generation with guidance.",
      "prerequisites": [
        "representation-learning-autoencoders-vaes-latent-spaces",
        "generative-adversarial-networks",
        "information-theory-core"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.16-GM4-generative-adding-noise.md",
      "code": "GM4",
      "number": "3.16",
      "frequency": 3,
      "prerequisiteTitles": [
        "2. Autoencoders and VAEs",
        "3. GAN",
        "Information Theory for AI"
      ]
    },
    {
      "id": "evaluating-generative-models",
      "title": "Evaluating Generative Models",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Use several measurements tied to the intended use.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.26-GM7-evaluating-generative-models.md",
      "code": "GM7",
      "number": "3.26",
      "prerequisiteTitles": []
    },
    {
      "id": "sciml-setup",
      "title": "1. The equation and the network",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "The heat equation, and a network whose output is its solution.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.19-XT3-sciml-setup.md",
      "code": "XT3",
      "number": "3.19",
      "frequency": 3,
      "prerequisiteTitles": []
    },
    {
      "id": "sciml-derivatives",
      "title": "2. Derivatives and the physics loss",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Derivatives of a network by autograd, and a loss built from the equation.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.22-XT6-sciml-derivatives.md",
      "code": "XT6",
      "number": "3.22",
      "frequency": 3,
      "prerequisiteTitles": []
    },
    {
      "id": "sciml-training-and-checking",
      "title": "3. Training and checking",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "L-BFGS, checking against a trusted solution, and inverse problems.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.24-XT8-sciml-training-and-checking.md",
      "code": "XT8",
      "number": "3.24",
      "frequency": 3,
      "prerequisiteTitles": []
    },
    {
      "id": "neural-operators",
      "title": "Neural operators: DeepONet and FNO",
      "section": "camp",
      "level": "Advanced",
      "time": "1 week",
      "summary": "A neural operator learns the solution operator mapping a coefficient function or initial condition to the solution.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/3_Camp/3.25-XT9-neural-operators.md",
      "code": "XT9",
      "number": "3.25",
      "prerequisiteTitles": []
    },
    {
      "id": "efficient-attention-long-context",
      "title": "Efficient Attention",
      "section": "advanced",
      "level": "Advanced",
      "time": "1-2 weeks",
      "summary": "Scale long-context attention with FlashAttention, linear, low-rank and hashing attention, and judge what each approximation costs.",
      "prerequisites": [
        "transformer-fundamentals",
        "programming-essentials",
        "convolutional-neural-networks",
        "kernel-methods-svms",
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
        "sequence-models-transfer-learning"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.3-TF9-efficient-attention-long-context.md",
      "code": "TF9",
      "number": "4.3",
      "prerequisiteTitles": [
        "Self-Attention Fundamentals",
        "NumPy Programming",
        "Convolutional Neural Networks",
        "Support Vector Machines",
        "Eigenvalues and Spectral Theory",
        "Recurrent Networks: RNN, LSTM, GRU"
      ]
    },
    {
      "id": "efficient-setup",
      "title": "1. Setup, prefill and decode",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Where long-context inference spends its time, and its two phases.",
      "prerequisites": [
        "transformer-fundamentals"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.7-TF25-efficient-setup.md",
      "code": "TF25",
      "number": "4.7",
      "frequency": 3,
      "prerequisiteTitles": [
        "Self-Attention Fundamentals"
      ]
    },
    {
      "id": "efficient-kv-cache",
      "title": "2. The KV cache and sharing it",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Keeping keys and values between steps, and sharing them across heads.",
      "prerequisites": [
        "efficient-setup",
        "transformer-fundamentals",
        "gpt-causal-lm",
        "relative-position-rope-positional-methods"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.9-TF27-efficient-kv-cache.md",
      "code": "TF27",
      "number": "4.9",
      "frequency": 3,
      "prerequisiteTitles": [
        "1. Setup, prefill and decode",
        "Self-Attention Fundamentals",
        "GPT and Causal Language Modeling",
        "Positional Encoding and RoPE"
      ]
    },
    {
      "id": "efficient-flashattention",
      "title": "3. FlashAttention and linear attention",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Exact attention in blocks, and attention with a running state instead of a softmax.",
      "prerequisites": [
        "transformer-fundamentals"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.11-TF29-efficient-flashattention.md",
      "code": "TF29",
      "number": "4.11",
      "frequency": 3,
      "prerequisiteTitles": [
        "Self-Attention Fundamentals"
      ]
    },
    {
      "id": "sparse-attention-sinks-and-eviction",
      "title": "4. Sparse attention, sinks and eviction",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Bounding memory means dropping cache entries, and which entries determine whether the model survives.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.13-TF31-sparse-attention-sinks-and-eviction.md",
      "code": "TF31",
      "number": "4.13",
      "prerequisiteTitles": []
    },
    {
      "id": "state-space-models",
      "title": "5. State-space models",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "A state-space model gets parallel training and constant-memory generation, because linearity lets the recurrence be rewritten as a convolution.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.14-TF32-state-space-models.md",
      "code": "TF32",
      "number": "4.14",
      "prerequisiteTitles": []
    },
    {
      "id": "mixture-of-experts",
      "title": "Mixture of Experts",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Sparse routing: growing parameter count without growing per-token compute.",
      "prerequisites": [
        "transformer-residual-and-layernorm",
        "neural-networks-perceptrons-mlps-activations"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.15-TF33-mixture-of-experts.md",
      "code": "TF33",
      "number": "4.15",
      "frequency": 0,
      "prerequisiteTitles": [
        "4. The Transformer block",
        "The Multilayer Perceptron"
      ]
    },
    {
      "id": "reinforcement-learning",
      "title": "Reinforcement Learning: MDPs and Q-Learning",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "MDPs, Bellman equations, and learning a policy from experience alone.",
      "prerequisites": [
        "expectations-multivariate-distributions",
        "neural-networks-perceptrons-mlps-activations"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.16-RF2-reinforcement-learning.md",
      "code": "RF2",
      "number": "4.16",
      "frequency": 0,
      "prerequisiteTitles": [
        "Expectation and Joint Distributions",
        "The Multilayer Perceptron"
      ]
    },
    {
      "id": "policy-gradients",
      "title": "Policy Gradients and Actor-Critic",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Optimizing a policy directly: REINFORCE, baselines, advantage, and PPO.",
      "prerequisites": [
        "reinforcement-learning",
        "backpropagation-mechanics"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.17-RF3-policy-gradients.md",
      "code": "RF3",
      "number": "4.17",
      "frequency": 0,
      "prerequisiteTitles": [
        "Reinforcement Learning: MDPs and Q-Learning",
        "Backpropagation by Hand"
      ]
    },
    {
      "id": "preference-optimization-rlhf-and-dpo",
      "title": "RLHF and Preference Optimization",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Optimize the policy against the reward model with PPO and a KL penalty to the reference, or replace the RL loop entirely with DPO.",
      "prerequisites": [],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.18-RF4-preference-optimization-rlhf-and-dpo.md",
      "code": "RF4",
      "number": "4.18",
      "prerequisiteTitles": []
    },
    {
      "id": "graph-neural-networks",
      "title": "Graph Neural Networks",
      "section": "advanced",
      "level": "Advanced",
      "time": "1 week",
      "summary": "Message passing, the GCN layer, and the symmetry it is built around.",
      "prerequisites": [
        "neural-networks-perceptrons-mlps-activations",
        "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory"
      ],
      "author": "USAAIO Guide Team",
      "file": "content/4_Advanced/4.6-XT2-graph-neural-networks.md",
      "code": "XT2",
      "number": "4.6",
      "frequency": 1,
      "prerequisiteTitles": [
        "The Multilayer Perceptron",
        "Eigenvalues and Spectral Theory"
      ]
    }
  ];

  window.AIO_CONTENT = { modules };
})();
