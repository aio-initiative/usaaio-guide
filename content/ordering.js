(function () {
  const SECTION_ORDERING = {
    "round1": {
      "label": "Round 1",
      "seoTitle": "Round 1 for USA AI Olympiad",
      "description": "Python, mathematics, probability, machine learning, and computer vision.",
      "path": "content/1_Round_1",
      "route": "round1/",
      "chapters": [
        {
          "name": "Python",
          "description": "",
          "items": [
            "python-basics",
            "programming-essentials",
            "pandas-data-cleaning-storing",
            "plotting-matplotlib-seaborn",
            "colab-markdown-notebooks",
            "scikit-learn-api-ml-pipelines",
            "why-pytorch-not-tensorflow",
            "pytorch-tensors-autograd",
            "training-loop-in-pytorch"
          ]
        },
        {
          "name": "Math",
          "description": "",
          "items": [
            "linear-algebra-i-vectors-matrices-systems-transformations",
            "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory",
            "calculus-gradients"
          ]
        },
        {
          "name": "Probability and Statistics",
          "description": "",
          "items": [
            "probability-foundations",
            "expectations-multivariate-distributions",
            "statistics-hypothesis-testing"
          ]
        },
        {
          "name": "Classical ML",
          "description": "",
          "items": [
            "data-processing-features",
            "linear-logistic-regression",
            "loss-functions",
            "kernel-methods-svms",
            "decision-trees-knn",
            "tree-based-models-ensembles",
            "clustering",
            "clustering-methods"
          ]
        },
        {
          "name": "Deep Learning",
          "description": "",
          "items": [
            "neural-networks-perceptrons-mlps-activations",
            "neural-network-regularization-dropout-batchnorm",
            "optimization-adam-sgd"
          ]
        },
        {
          "name": "Computer Vision",
          "description": "",
          "items": [
            "convolutional-neural-networks"
          ]
        }
      ]
    },
    "round2": {
      "label": "Round 2",
      "seoTitle": "Round 2 for AIO",
      "description": "Backpropagation, model evaluation, computer vision, and transformers.",
      "path": "content/2_Round_2",
      "route": "round2/",
      "chapters": [
        {
          "name": "Math",
          "description": "",
          "items": [
            "forward-propagation",
            "backpropagation-mechanics"
          ]
        },
        {
          "name": "Classical ML",
          "description": "",
          "items": [
            "model-evaluation",
            "bias-variance-cross-validation",
            "principal-component-analysis",
            "nonlinear-dimensionality-reduction",
            "semi-supervised-learning",
            "time-series"
          ]
        },
        {
          "name": "Deep Learning",
          "description": "",
          "items": [
            "advanced-optimization",
            "advanced-optimization-methods"
          ]
        },
        {
          "name": "Computer Vision",
          "description": "",
          "items": [
            "vision-setup",
            "vision-convolution",
            "vision-classify",
            "hierarchical-vision-models",
            "semantic-segmentation-unet-masks",
            "modern-vision-architectures-resnet-efficientnet-transfer"
          ]
        },
        {
          "name": "Transformers and Attention",
          "description": "",
          "items": [
            "transformer-setup",
            "transformer-qkv",
            "transformer-weighted-sum",
            "transformer-residual-and-layernorm",
            "transformer-predict",
            "transformer-fundamentals",
            "transformer-architectures-bert-mlm",
            "gpt-causal-lm",
            "transformer-architectures-t5-encoder-decoder"
          ]
        },
        {
          "name": "Representation Learning",
          "description": "",
          "items": [
            "contrastive-learning"
          ]
        }
      ]
    },
    "camp": {
      "label": "Camp",
      "seoTitle": "AIO Camp Preparation",
      "description": "Information theory, vision, NLP, generative models, and representation learning.",
      "path": "content/3_Camp",
      "route": "camp/",
      "chapters": [
        {
          "name": "Probability and Statistics",
          "description": "",
          "items": [
            "information-theory-core"
          ]
        },
        {
          "name": "Computer Vision",
          "description": "",
          "items": [
            "vision-transformers-vit-attention"
          ]
        },
        {
          "name": "NLP",
          "description": "",
          "items": [
            "natural-language-processing-tokenization-embeddings",
            "pre-training-fine-tuning",
            "sequence-models-transfer-learning"
          ]
        },
        {
          "name": "Transformers and Attention",
          "description": "",
          "items": [
            "rag-retrieval-embeddings",
            "relative-position-rope-positional-methods",
            "cross-attention-multimodal-attention"
          ]
        },
        {
          "name": "Representation Learning",
          "description": "",
          "items": [
            "representation-learning-autoencoders-vaes-latent-spaces",
            "autoregressive-generative-models"
          ]
        },
        {
          "name": "Generative Models",
          "description": "",
          "items": [
            "generative-setup",
            "generative-adversarial-networks",
            "generative-adding-noise",
            "evaluating-generative-models"
          ]
        },
        {
          "name": "Extra",
          "description": "",
          "items": [
            "sciml-setup",
            "sciml-derivatives",
            "sciml-training-and-checking",
            "neural-operators"
          ]
        }
      ]
    },
    "advanced": {
      "label": "Advanced",
      "seoTitle": "Advanced AIO Topics",
      "description": "Long-context attention, multimodal models, reinforcement learning, and graph networks.",
      "path": "content/4_Advanced",
      "route": "advanced/",
      "chapters": [
        {
          "name": "Transformers and Attention",
          "description": "",
          "items": [
            "efficient-attention-long-context",
            "efficient-setup",
            "efficient-kv-cache",
            "efficient-flashattention",
            "sparse-attention-sinks-and-eviction",
            "state-space-models",
            "mixture-of-experts"
          ]
        },
        {
          "name": "Representation Learning",
          "description": "",
          "items": []
        },
        {
          "name": "Reinforcement Learning",
          "description": "",
          "items": [
            "reinforcement-learning",
            "policy-gradients",
            "preference-optimization-rlhf-and-dpo"
          ]
        },
        {
          "name": "Extra",
          "description": "",
          "items": [
            "graph-neural-networks"
          ]
        }
      ]
    }
  };

  const sectionIds = Object.keys(SECTION_ORDERING);
  const moduleIdToSectionMap = {};
  const moduleIdToURLMap = {};

  sectionIds.forEach((sectionId) => {
    SECTION_ORDERING[sectionId].chapters.forEach((chapter) => {
      chapter.items.forEach((moduleId) => {
        moduleIdToSectionMap[moduleId] = sectionId;
        moduleIdToURLMap[moduleId] = `/module/?id=${moduleId}`;
      });
    });
  });

  window.AIO_ORDERING = {
    sections: sectionIds,
    sectionOrdering: SECTION_ORDERING,
    moduleIdToSectionMap,
    moduleIdToURLMap,
  };
})();
