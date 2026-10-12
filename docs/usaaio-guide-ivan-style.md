# USAAIO Guide — Ivan-style condensed edition

This is a style-rewritten, condensed single-file edition of the source guide. It keeps the guide’s order and the ideas worth carrying around, while folding repeated library instructions and routine practice lists into the explanations. Each section starts from a problem, follows the first attempt, and explains why the next method is needed.

## Arc I — Foundations: getting data and derivatives to behave

### 1.1 · PY1 — NumPy Programming

Suppose three students have scores in three subjects, so the data is naturally a \(3\times3\) array. The two axes have different meanings: axis \(0\) indexes students and axis \(1\) indexes subjects. Calling `mean(axis=1)` therefore returns one mean per student. NumPy can run this reduction in compiled code because every entry has a common dtype and the array is stored with a known stride pattern.

Broadcasting is a rule for matching shapes. Starting from the trailing dimension, two dimensions are compatible when they are equal or one of them is \(1\). Thus an array of shape \((1,3)\) and one of shape \((2,1)\) broadcast to \((2,3)\). The implementation can reuse an address along a stride-zero dimension instead of allocating repeated values. Basic slicing usually returns a view because the result can still be described by an offset, shape, and strides into the same buffer. Integer or Boolean indexing has to gather arbitrary entries, so it usually allocates a new array. This distinction determines whether assigning through the result can mutate the original.

Pairwise squared distances often use

$$
\lVert x-y\rVert^2=\lVert x\rVert^2+\lVert y\rVert^2-2x^\top y,
$$

For matrices \(X\in\mathbb R^{m\times d}\) and \(Y\in\mathbb R^{n\times d}\), this gives all \(mn\) distances from the two norm vectors and \(XY^\top\). The rearrangement is algebraically exact over the reals. In floating-point arithmetic each operation is rounded, so the two evaluation orders need not agree. When \(x\) and \(y\) are close, the expanded form subtracts nearly equal large terms and can lose most significant digits. That is a numerical reason to prefer a stable library routine when the distances are small relative to the coordinates.

### 1.2 · PY2 — Pandas Data Cleaning

The same measurements become a DataFrame: rows are observations, columns are variables, and the index is a label that pandas will use for alignment. That last feature is powerful and dangerous; adding two Series with the same values in different index orders can pair the wrong-looking rows while producing no shape error. Use `loc` for labels, `iloc` for positions, boolean masks for conditions, and do not treat a missing value as zero unless the data-generating process says “unobserved” really means zero. Cleaning is already modelling: imputing income, collapsing rare categories, parsing dates, removing duplicates, and deciding which table keys define a person all encode assumptions. Long tables are convenient for grouped analysis and plotting; wide tables are convenient for some reports. `groupby` is split–apply–combine, joins require an explicit key and a check for unexpected multiplicity, and an EDA pass should inspect missingness, cardinality, skew, outliers, duplicates, and target leakage before a model gets involved.

### 1.3 · PY3 — Data Visualization

A plot is useful when it answers a question. A histogram shows how one variable is distributed, a scatter plot shows how two variables move together, a line plot shows how an ordered quantity changes, and a residual plot shows what the model systematically missed. Matplotlib’s figure contains axes, and axes contain artists such as lines, patches, and text; using that object-oriented vocabulary keeps several plots from accidentally sharing state. Seaborn adds sensible statistical mappings for hue, style, and faceting, but it cannot make a biased sample representative. Check bin choices, smoothing, overplotting, color accessibility, and whether a truncated axis exaggerates a difference. A deliberately shortened y-axis can make two nearly equal bars look wildly different. For modelling, plot train and validation curves, class balance, feature distributions before and after transforms, and predictions against residuals. Decide what conclusion the plot is meant to support before choosing the plotting call.

### 1.4 · PY4 — Colab and Markdown Notebooks

A notebook has two orders: the visible order of cells and the execution order stored in the kernel. Those can disagree. A later cell may read a variable created by a cell that now appears below it, or may use an old value after its defining cell was edited. Restarting the kernel deletes that hidden state; running top to bottom then checks whether the visible program is sufficient to reproduce the result. If it fails, the immediate conclusion is only that the notebook depended on state not created by that run. The cause could be an out-of-order cell, an undeclared file, an unrecorded package version, randomness, or an external service. Put installation, imports, seeds, and configuration near the top, make file paths explicit, and record the versions that affect the result. Markdown should state what a cell is testing and how its output bears on the conclusion, so another reader can distinguish an experiment from leftover exploration.

### 1.5 · PY5 — Scikit-Learn Pipelines

Scikit-learn estimators store quantities learned by `fit` and reuse them in `transform`, `predict`, or `predict_proba`. A `Pipeline` makes those fitted transformations part of one estimator, while `ColumnTransformer` applies different transformations to selected columns. Suppose standardization uses \(\hat\mu\) and \(\hat\sigma\). If those values are computed from all rows before cross-validation, each validation fold affects the representation of its own training fold. The reported score is then conditional on information that would not exist for a genuinely unseen sample. Fitting the entire pipeline inside each training fold gives fold-specific \(\hat\mu_{\text{train}}\), \(\hat\sigma_{\text{train}}\), category vocabularies, and imputers. The split itself must match the independent deployment unit: group repeated measurements from one patient, and use a temporal split when prediction is about the future.

### 1.6 · PY6 — PyTorch Fundamentals

PyTorch is the tensor and automatic-differentiation version of the same data-to-objective workflow. Tensors have shape, dtype, device, and sometimes a computation history; `nn.Module` holds parameters, a `Dataset` describes examples, and a `DataLoader` batches them. A training loop repeatedly runs forward, computes a scalar loss, clears old gradients, calls `backward`, and lets the optimizer update parameters; evaluation changes dropout and batch-normalization behaviour through `eval()`. Autograd records elementary operations so reverse-mode differentiation can walk from the scalar loss to every parameter. For (f(a,b)=\tanh(ab+a^2)), with (a=2,b=-3), the inner value is (-2), and

$$
\frac{\partial f}{\partial a}=(1-f^2)(b+2a),\qquad \frac{\partial f}{\partial b}=(1-f^2)a.
$$

Hand-checking this once is more educational than memorizing `loss.backward()`. Device mismatches, missing `zero_grad`, retained graphs, and wrong batch shapes are the ordinary failure modes.

### 1.7 · MA1 — Vectors, Matrices, Systems, Transformations

A vector is an element of a vector space: it can be added to another vector and multiplied by a scalar. A point belongs to an affine space, where subtracting two points produces a vector but adding two points is not intrinsically defined. Once an origin and coordinates are chosen, both are represented by tuples in \(\mathbb R^n\), which is why elementary treatments often blur the distinction. The dot product is a bilinear form \(x^\top y\); it induces the norm \(\lVert x\rVert_2=\sqrt{x^\top x}\), and the distance between points \(p,q\) is the norm of the displacement \(p-q\). The orthogonal projection of \(x\) onto the one-dimensional subspace spanned by a nonzero \(u\) is

$$
\operatorname{proj}_u(x)=\frac{x^\top u}{u^\top u}u,
$$

because the coefficient \(\alpha\) minimizing \(\lVert x-\alpha u\rVert_2^2\) satisfies

$$
0=\frac{d}{d\alpha}\lVert x-\alpha u\rVert_2^2=-2u^\top x+2\alpha u^\top u.
$$

Consequently \(u^\top(x-\operatorname{proj}_u x)=0\). The same normal-equation argument gives projection onto a column space: minimizing \(\lVert Ax-b\rVert_2^2\) yields \(A^\top(Ax-b)=0\). A matrix \(A\) represents a linear map after bases have been chosen. Its \(j\)-th column is the image of the \(j\)-th basis vector, its rank is the dimension of its image, and \(Ax=b\) is solvable exactly when \(b\) belongs to that image. Matrix multiplication represents composition, so \(AB\) and \(BA\) generally differ because the two compositions apply maps in opposite orders.

### 1.8 · MA2 — Eigenvalues and Spectral Theory

An eigenpair satisfies \(Av=\lambda v\) with \(v\ne0\). If \(A=V\Lambda V^{-1}\), then \(A^t=V\Lambda^tV^{-1}\), so repeated application is controlled by the eigenvalues. For

$$
A=\begin{bmatrix}2&1\\1&2\end{bmatrix},
$$

\((1,1)^\top\) and \((1,-1)^\top\) are eigenvectors with eigenvalues \(3\) and \(1\). A real symmetric matrix has an orthonormal eigendecomposition \(A=Q\Lambda Q^\top\). Writing \(x=Qz\) gives

$$
x^\top Ax=z^\top\Lambda z=\sum_i\lambda_i z_i^2.
$$

Therefore \(A\) is positive definite exactly when every eigenvalue is positive. The trace and determinant identities follow from similarity invariance and the diagonal form: \(\operatorname{tr}(A)=\sum_i\lambda_i\) and \(\det(A)=\prod_i\lambda_i\), counting algebraic multiplicity. Covariance matrices, graph Laplacians, and Hessians are symmetric, so their spectra respectively describe variance directions, graph-frequency modes, and local second-order curvature.

### 1.9 · MA3 — Calculus and Gradients

For differentiable \(f:\mathbb R^n\to\mathbb R\), the gradient is defined by the first-order expansion

$$
f(x+\Delta)=f(x)+\nabla f(x)^\top\Delta+o(\lVert\Delta\rVert).
$$

Among unit vectors \(u\), Cauchy–Schwarz gives \(\nabla f(x)^\top u\ge-\lVert\nabla f(x)\rVert\), with equality at \(u=-\nabla f/\lVert\nabla f\rVert\). This is the precise sense in which the negative gradient is the steepest local descent direction. For a vector-valued map, the Jacobian is its linear first-order approximation; for scalar \(f\), the Hessian is the derivative of the gradient.

For least squares,

$$
L(w)=\lVert y-Xw\rVert^2
\quad\Longrightarrow\quad
\nabla_wL=2X^\top(Xw-y).
$$

This follows by expanding the quadratic or taking the differential \(dL=2(Xw-y)^\top X\,dw\). For logits \(z\), softmax probabilities \(p_i=e^{z_i}/\sum_je^{z_j}\), and one-hot target \(y\), cross-entropy is

$$
L=-\sum_i y_i\log p_i=-y^\top z+\log\sum_j e^{z_j}.
$$

Differentiating gives \(\partial L/\partial z_i=p_i-y_i\). The simplification comes from differentiating the log-normalizer; it is not a general rule that losses produce “prediction minus target.” A centered finite-difference check,

$$
\frac{L(w+he_i)-L(w-he_i)}{2h},
$$

has truncation error \(O(h^2)\), while roundoff grows when \(h\) is too small. It is useful for checking a few derivatives but costs two forward evaluations per coordinate.

## Arc II — Probability and classical ML: deciding what the data means

### 1.10 · PS1 — Probability and Bayes

Suppose disease prevalence is \(P(D)=0.001\), sensitivity is \(P(+\mid D)=0.99\), and specificity is \(P(-\mid D^c)=0.99\). Then

$$
P(D\mid+)
=\frac{0.99(0.001)}
{0.99(0.001)+0.01(0.999)}
\approx0.090.
$$

The positive predictive value is about \(9\%\), despite both sensitivity and specificity being \(99\%\), because false positives are drawn from a much larger group. In general, conditional probability is \(P(A\mid B)=P(A\cap B)/P(B)\), and Bayes’ theorem follows by writing the same joint probability in two ways:

$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.
$$

In a parametric model, the prior is a density \(p(\theta)\), the likelihood is the same function \(p(x\mid\theta)\) viewed as a function of \(\theta\) for fixed observed \(x\), and the posterior is

$$
p(\theta\mid x)=\frac{p(x\mid\theta)p(\theta)}
{\int p(x\mid\vartheta)p(\vartheta)\,d\vartheta}.
$$

Independence of \(X\) and \(Y\) means their joint distribution factors as \(p_{XY}(x,y)=p_X(x)p_Y(y)\); zero covariance is only one scalar consequence of this factorization when second moments exist. Bernoulli, binomial, categorical, and Gaussian distributions are probability models with different sample spaces and assumptions. Their expectations and variances are computed from the definitions in the next section rather than serving as informal synonyms for “average” and “spread.”

### 1.11 · PS2 — Expectation and Multivariate Distributions

For an integrable random variable \(X\), expectation is the probability-weighted integral

$$
\mathbb E[X]=\int_\Omega X(\omega)\,dP(\omega).
$$

Linearity follows from linearity of the integral, so \(\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y]\) whether or not \(X\) and \(Y\) are independent. Variance is the second central moment,

$$
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2]
=\mathbb E[X^2]-(\mathbb E X)^2.
$$

Expanding the square gives the formula that is usually hidden behind the phrase “variance adds”:

$$
\operatorname{Var}\!\left(\sum_{i=1}^n X_i\right)
=\sum_{i=1}^n\operatorname{Var}(X_i)
+2\sum_{i<j}\operatorname{Cov}(X_i,X_j).
$$

The covariance terms vanish for pairwise independent variables, although zero covariance by itself does not imply independence. Hence for iid variables with variance \(\sigma^2\),

$$
\operatorname{Var}(\bar X)=\operatorname{Var}\!\left(\frac1n\sum_iX_i\right)=\frac{\sigma^2}{n}.
$$

For a random vector \(X\in\mathbb R^d\), the covariance matrix is \(\Sigma=\mathbb E[(X-\mu)(X-\mu)^\top]\), and \(a^\top\Sigma a=\operatorname{Var}(a^\top X)\ge0\), so every covariance matrix is positive semidefinite. A multivariate Gaussian has log density proportional to \(-(x-\mu)^\top\Sigma^{-1}(x-\mu)/2\); its equal-density sets are therefore ellipsoids whose principal axes are the eigenvectors of \(\Sigma\) and whose squared radii scale with its eigenvalues.

### 1.12 · PS3 — Statistical Inference and Hypothesis Testing

An estimator \(\hat\theta=T(X_1,\ldots,X_n)\) is itself a random variable because it depends on the sample. Its bias and variance are

$$
\operatorname{Bias}(\hat\theta)=\mathbb E[\hat\theta]-\theta,\qquad
\operatorname{Var}(\hat\theta)=\mathbb E[(\hat\theta-\mathbb E\hat\theta)^2].
$$

For squared error these combine exactly:

$$
\mathbb E[(\hat\theta-\theta)^2]
=\operatorname{Var}(\hat\theta)+\operatorname{Bias}(\hat\theta)^2.
$$

The denominator \(n-1\) in the sample variance is not justified merely by saying that the mean “uses a degree of freedom.” Let \(X_i=\mu+\varepsilon_i\), where the iid errors have mean \(0\) and variance \(\sigma^2\). Since

$$
\sum_i(X_i-\bar X)^2=\sum_i(X_i-\mu)^2-n(\bar X-\mu)^2,
$$

taking expectations gives \(n\sigma^2-n(\sigma^2/n)=(n-1)\sigma^2\). Therefore \(S^2=\frac1{n-1}\sum_i(X_i-\bar X)^2\) is unbiased for \(\sigma^2\).

A \(95\%\) confidence interval is a random set \(C(X)\) constructed so that \(P_\theta(\theta\in C(X))\ge0.95\) under the stated model. After observing the data, a frequentist parameter is fixed and the interval either contains it or does not. A p-value is \(P_{H_0}(T(X)\ge T(x_{\mathrm{obs}}))\), or the corresponding two-sided tail, for a test statistic chosen in advance. Its interpretation depends on the null model, sampling design, and stopping rule; it is not \(P(H_0\mid X)\). The same sampling logic applies to validation scores: a reported score is an estimator whose uncertainty depends on the number of independent test units and the split procedure.

### 1.13 · ML1 — Data Processing and Features

Before choosing a model, decide what each feature represents. Missing numerical values need an imputation rule; categories need an encoding whose order is justified; skewed incomes may benefit from a log transform; dates may need cyclic or elapsed-time features. Target encoding is powerful precisely because it uses the target, so it must be fitted inside the training fold or it leaks. Standardization helps distance-based and gradient-based methods when units are arbitrary, but it can destroy the meaning of a physically calibrated quantity. Split by the unit that must generalize: by patient, household, user, or time rather than by row if rows share hidden information. A feature is a hypothesis about the data-generating process. Cleaning, visualization, and pipeline construction determine what information the model is allowed to use. kNN and SVM make this especially visible because the metric and scale are part of the model itself.

### 1.14 · ML2 — Supervised Learning Foundations

Linear regression projects a target onto the column space of (X), with normal equations (X^\top Xw=X^\top y) when the inverse is well behaved; ridge adds (\lambda\lVert w\rVert^2) so unstable directions are shrunk instead of trusted. Polynomial features make a linear model nonlinear in the original input while remaining linear in its expanded features. Logistic regression makes log-odds linear, (\log[p/(1-p)]=w^\top x+b), and maximizes a likelihood equivalent to minimizing cross-entropy; the objective is convex but usually has no ordinary closed form. Perfect separation can send coefficients toward infinity, and a threshold such as (0.5) is a decision choice, not a law of probability. kNN stores examples, uses a distance metric at prediction time, and trades bias against variance through (k). In high dimensions distances concentrate, so “nearest” becomes less meaningful. Compare training and held-out behaviour; a flexible model can fit the first and perform worse on the second.

### 1.15 · ML3 — Kernel Methods and SVMs

For labels \(y_i\in\{-1,1\}\), a separating hyperplane satisfies \(y_i(w^\top x_i+b)>0\). Rescaling \(w,b\) does not change the boundary, so impose \(y_i(w^\top x_i+b)\ge1\). The distance from \(x_i\) to the boundary is \(|w^\top x_i+b|/\lVert w\rVert\), making the distance between the two supporting hyperplanes \(2/\lVert w\rVert\). Maximizing the margin is therefore

$$
\min_{w,b}\frac12\lVert w\rVert^2
\quad\text{subject to}\quad y_i(w^\top x_i+b)\ge1.
$$

Soft margins introduce \(\xi_i\ge0\) with \(y_i(w^\top x_i+b)\ge1-\xi_i\), producing the equivalent regularized hinge-loss objective \(\frac12\lVert w\rVert^2+C\sum_i\max(0,1-y_if(x_i))\). The Lagrangian dual depends on the data only through \(x_i^\top x_j\). Replacing this inner product by \(K(x_i,x_j)\) is valid when every finite Gram matrix is positive semidefinite; then \(K\) is an inner product \(\langle\phi(x_i),\phi(x_j)\rangle\) in some feature space. Polynomial and RBF kernels define different spaces, while \(C\), feature scaling, and kernel parameters determine the effective regularization.

### 1.16 · ML4 — Tree-Based Models and Ensembles

A decision tree chooses a split by the reduction in an impurity functional. For classification, Gini impurity at a node with class proportions \(p_k\) is \(1-\sum_kp_k^2\); for regression, minimizing squared error in a leaf makes the optimal leaf prediction its sample mean. A high-cardinality identifier offers many candidate partitions and can achieve a large in-sample impurity reduction by chance, so minimum leaf sizes, pruning, and held-out evaluation are necessary.

The variance reduction from bagging can be stated exactly. Suppose \(T_1,\ldots,T_M\) have common variance \(\sigma^2\) and pairwise correlation \(\rho\). Then

$$
\operatorname{Var}\!\left(\frac1M\sum_{m=1}^M T_m\right)
=\rho\sigma^2+\frac{1-\rho}{M}\sigma^2.
$$

Increasing \(M\) removes the uncorrelated part but leaves the correlated part. Random forests sample features at each split largely to reduce \(\rho\), even if an individual tree becomes slightly weaker. Boosting is different: it adds functions sequentially in a direction that reduces the current empirical loss, with gradient boosting fitting each new tree to a negative functional gradient or a related residual. Split-count feature importance measures how often a training criterion improved; correlated predictors can substitute for one another, so it is neither a causal quantity nor generally stable.

### 1.17 · ML5 — Clustering

Unsupervised learning begins with an uncomfortable question: what does “cluster” mean for this dataset? k-means chooses assignments and centroids to minimize

$$
\sum_i\lVert x_i-\mu_{c_i}\rVert^2,
$$

then alternates assignment and mean updates; k-means++ improves initialization, but the method still prefers roughly round, similarly scaled groups. Agglomerative clustering builds a hierarchy whose result depends on single, complete, average, or Ward linkage. Gaussian mixtures replace hard labels with probabilities and fit them by EM, although a component covariance can collapse onto one point unless regularized. DBSCAN finds dense core points, border points, and noise without choosing (K), but its radius and minimum-neighbour parameters fail when density varies. k-means is a special hard-assignment, equal-covariance mixture model. For any clustering method, check stability, interpretability, and whether the grouping helps with the downstream question.

### 2.3 · ML6 — Model Evaluation

Evaluation is the experiment around the model. Separate training, validation, and final test roles; if hyperparameters are selected, nested cross-validation gives a less optimistic estimate. For classification, a confusion matrix leads to precision, recall, and F1, while ROC curves can look flattering on severe class imbalance where precision–recall curves tell the more relevant story. A probability can be well-ranked but badly calibrated, so inspect reliability and use proper scoring rules such as log loss or the Brier score. Thresholds should reflect costs, not be inherited from habit. Regression metrics answer different questions: MSE heavily punishes large errors, MAE is more robust, and (R^2) can be negative out of sample and says nothing about calibration. Learning curves expose whether more data or more capacity is likely to help. Report uncertainty, split protocol, preprocessing, and the number of examples; a score without its measurement procedure is not a result.

### 2.4 · ML7 — Dimensionality Reduction

Let the rows of \(X\in\mathbb R^{n\times d}\) be centered observations and let \(v\) be a unit vector. The variance of the one-dimensional coordinates \(Xv\) is

$$
\frac1n\lVert Xv\rVert_2^2
=v^\top\!\left(\frac1nX^\top X\right)v.
$$

Maximizing this Rayleigh quotient subject to \(v^\top v=1\) gives the leading eigenvector of the sample covariance matrix. Subsequent components maximize the same quantity subject to orthogonality, so \(V_k\) contains the top \(k\) eigenvectors. The reconstruction from the projected coordinates is \(XV_kV_k^\top\). Because

$$
\lVert X-XV_kV_k^\top\rVert_F^2
=\lVert X\rVert_F^2-\lVert XV_k\rVert_F^2,
$$

maximizing retained variance is exactly the same optimization as minimizing squared reconstruction error. If \(X=U\Sigma V^\top\), these directions are the first \(k\) columns of \(V\), and the discarded error is \(\sum_{j>k}\sigma_j^2\). None of this says that the high-variance directions predict the target; PCA is unsupervised. Kernel PCA replaces inner products with a centered kernel matrix. MDS, Isomap, LLE, t-SNE, UMAP, and random projection preserve different quantities, so distances and cluster sizes in a two-dimensional output are meaningful only to the extent guaranteed by that method’s objective.

## Arc III — Deep learning and vision: learning the right transformations

### 1.18 · DL1 — Neural Networks, Perceptrons, MLPs, and Activations

Take a toy XOR dataset. One affine classifier cannot separate it, and stacking affine maps without a nonlinearity does not help because (W_2(W_1x+b_1)+b_2) is still one affine map. A multilayer perceptron becomes expressive when an activation breaks that collapse: ReLU keeps positive slopes and can create dead units, sigmoid and tanh saturate, and GELU smooths the gating idea. A layer computes (z=Wx+b), applies (h=\phi(z)), and passes the result onward. Initialization keeps activations and gradients at reasonable scale; universal approximation says a sufficiently wide network can represent broad function classes, not that gradient descent will find a useful representation quickly. The training loop is the chain rule from the calculus section repeated over batches. Calculate one two-layer forward pass by hand once. The model is a parameterized composition whose useful geometry comes from depth plus nonlinearity.

### 1.19 · DL2 — Regularization and BatchNorm

Regularization changes either the objective, the hypothesis class, or the data distribution seen in training. An \(\ell_2\)-regularized objective \(L(\theta)+\frac{\lambda}{2}\lVert\theta\rVert_2^2\) contributes the gradient \(\lambda\theta\); under plain SGD this multiplies the current parameter by \(1-\eta\lambda\) before the loss-gradient step. An \(\ell_1\) penalty has subgradient \(\lambda\,\operatorname{sign}(\theta_j)\) away from zero and can produce exact zeros under proximal updates. Dropout samples a Bernoulli mask \(m\) and uses \(h'=m\odot h/(1-p)\), so \(\mathbb E[h'\mid h]=h\); training minimizes an expectation over these masks rather than the loss of one fixed network. Data augmentation similarly minimizes \(\mathbb E_{g\sim\mathcal G}L(f(gx),y)\), making the chosen transformation distribution part of the learning problem.

For a mini-batch \(B\), BatchNorm computes

$$
\hat h_i=\frac{h_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad
y_i=\gamma\hat h_i+\beta.
$$

Its output for one example therefore depends on the other examples in the batch. Evaluation uses running estimates of the population moments, which is why forgetting `eval()` changes predictions. LayerNorm instead computes moments across the features of each example, so it does not depend on batch composition. BatchNorm can add noise that has a regularizing effect, but its defining operation is normalization and reparameterization. Training and validation curves help separate optimization failure from excess fitting, although the diagnosis still depends on whether the validation split represents the intended deployment distribution.

### 1.20 · DL3 — Model Training and Generalization

The population risk is \(R(f)=\mathbb E_{(X,Y)\sim P}[\ell(f(X),Y)]\), while training minimizes the empirical risk \(\hat R_n(f)=\frac1n\sum_i\ell(f(x_i),y_i)\). Generalization concerns the gap \(R(f)-\hat R_n(f)\). A held-out average estimates \(R(f)\) only when its examples are drawn from the deployment distribution and are independent at the unit that matters. Repeated images of one patient cannot be split as if they were independent patients, and a random split of time-series rows estimates interpolation rather than future performance.

Optimization and generalization are distinct. A high training loss indicates that the optimizer, representation, or model class has not fit the training problem. A low training loss with substantially higher validation loss is evidence of a gap, but the cause may be finite-sample fitting, leakage in one split, or distribution shift. Learning-rate schedules alter the numerical trajectory: warmup limits early steps before moment estimates and activation scales stabilize; decay reduces the distance travelled by late noisy gradients. Batch size changes the covariance of the stochastic-gradient estimate, approximately shrinking it by \(1/B\) only when examples in a batch are independent. Keep the final test set untouched until model and protocol choices are fixed, otherwise adaptation to repeated test feedback makes it another validation set.

### 1.21 · DL4 — Optimization: Adam and SGD

SGD updates (\theta\leftarrow\theta-\eta g), momentum averages directions so a noisy valley has persistence, RMSProp rescales coordinates by recent squared gradients, and Adam combines both:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta\leftarrow\theta-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

If \(g_t\) has a stationary mean \(m\) and \(m_0=0\), then

$$
\mathbb E[m_t]=(1-\beta_1)\sum_{s=1}^t\beta_1^{t-s}m
=(1-\beta_1^t)m.
$$

Dividing by \(1-\beta_1^t\) removes this initialization bias; the same calculation applies to \(v_t\). AdamW updates \(\theta\) by a direct decay term \(-\eta\lambda\theta\) in addition to the adaptive gradient step. If an \(\ell_2\) penalty is instead added to \(g_t\), its \(\lambda\theta\) contribution enters the moment estimates and is divided coordinatewise by \(\sqrt{\hat v_t}+\epsilon\). The two procedures coincide for plain SGD up to parameterization, but not for an adaptive optimizer.

### 1.22 · CV1 — Convolutional Neural Networks

For a one-dimensional signal, the cross-correlation used by most deep-learning libraries is

$$
(K\star x)[i]=\sum_{u=0}^{k-1}K[u]\,x[i+u].
$$

The same kernel \(K\) is used at every valid \(i\). If \(T_a\) translates the input by \(a\), then away from boundary effects

$$
K\star(T_ax)=T_a(K\star x).
$$

This commutation relation is translation equivariance: translating the input translates the feature map. It is not invariance, which would require the output itself to remain unchanged. Pooling, global averaging, data augmentation, and later layers can make a classifier less sensitive to translations, but convolution alone does not.

For input length \(n\), padding \(p\), kernel size \(k\), and stride \(s\), the kernel’s left edge can start at \(0,s,2s,\ldots\) while its right edge remains within the padded length \(n+2p\). Thus the number of positions is

$$
\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1.
$$

In two dimensions the calculation applies to each spatial axis. A dense map from \(n\times n\) pixels to \(m\) outputs has \(O(n^2m)\) independent weights; a convolution with \(c_{\mathrm{in}}\) input channels, \(c_{\mathrm{out}}\) output channels, and a \(k\times k\) kernel has \(k^2c_{\mathrm{in}}c_{\mathrm{out}}\) weights regardless of image size. Local connectivity and shared parameters are the actual assumptions responsible for the difference.

### 2.1 · MA4 — Backpropagation Mechanics

Scalar chain rules become more useful, not less, in matrix form. If (z=Wx+b), an upstream vector (\delta=\partial L/\partial z) gives (\partial L/\partial W=\delta x^\top), (\partial L/\partial b=\delta), and (\partial L/\partial x=W^\top\delta). The transposes are forced by shapes and by the definition of a Jacobian–vector or vector–Jacobian product. Reverse mode is efficient when many parameters feed one scalar loss; it propagates one adjoint through the graph instead of forming every full Jacobian. Least squares, logistic loss, and softmax cross-entropy all simplify because their derivatives match the residual between prediction and target. Check a small network with finite differences, but choose (h) neither so large that curvature dominates nor so small that floating-point cancellation dominates. Autograd libraries are implementing this bookkeeping, not discovering a new calculus. If a gradient has the wrong shape, that is often the most useful clue.

### 2.2 · MA5 — Advanced Linear Algebra: Projections, SVD, and Low Rank

Every \(A\in\mathbb R^{m\times n}\) has an SVD \(A=U\Sigma V^\top\), where the columns of \(V\) are orthonormal eigenvectors of \(A^\top A\), \(\sigma_i=\sqrt{\lambda_i(A^\top A)}\), and \(u_i=Av_i/\sigma_i\) for nonzero \(\sigma_i\). Hence \(Av_i=\sigma_i u_i\): the right singular vectors form orthogonal input directions and the singular values state how much \(A\) scales each one. The rank is the number of nonzero singular values. The truncated matrix

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^\top
$$

minimizes both \(\lVert A-B\rVert_2\) and \(\lVert A-B\rVert_F\) over matrices \(B\) of rank at most \(k\), with errors \(\sigma_{k+1}\) and \((\sum_{i>k}\sigma_i^2)^{1/2}\), respectively. The pseudoinverse \(A^+=V\Sigma^+U^\top\) divides by nonzero singular values. A perturbation in a left-singular direction \(u_i\) is therefore amplified by \(1/\sigma_i\), explaining instability when \(\sigma_i\) is small. Forming \(A^\top A\) squares the spectral condition number because \(\kappa_2(A^\top A)=\kappa_2(A)^2\); QR or SVD avoids that additional loss of conditioning in least squares. For centered data, PCA uses the same right singular vectors because the covariance matrix is \(X^\top X/n\).

### 2.5 · DL5 — Advanced Optimization

A differentiable function is convex when

$$
f(y)\ge f(x)+\nabla f(x)^\top(y-x)
$$

for every \(x,y\). This first-order inequality implies that any point with \(\nabla f(x)=0\) is a global minimizer. \(L\)-smoothness means

$$
\lVert\nabla f(x)-\nabla f(y)\rVert\le L\lVert x-y\rVert,
$$

which implies the descent lemma

$$
f(x-\eta\nabla f(x))
\le f(x)-\eta\!\left(1-\frac{L\eta}{2}\right)\lVert\nabla f(x)\rVert^2.
$$

Thus \(0<\eta<2/L\) guarantees a decrease unless the gradient is zero. If \(f\) is also \(\mu\)-strongly convex, then \(f(y)\) is at least the tangent plane plus \(\frac{\mu}{2}\lVert y-x\rVert^2\). With \(\eta=1/L\), gradient descent satisfies

$$
f(x_t)-f(x^*)\le\left(1-\frac{\mu}{L}\right)^t
[f(x_0)-f(x^*)].
$$

The rate in this bound is controlled by the condition number \(L/\mu\). Newton’s step solves \(\nabla^2f(x)\Delta=-\nabla f(x)\); near a minimizer with a positive-definite, regular Hessian it can converge quadratically, but away from that regime the Hessian can be indefinite and a line search or trust region is needed. For an unbiased stochastic gradient \(g_t\) with nonzero covariance, a constant step size leaves a stationary error scale, while decreasing step sizes can converge under conditions such as \(\sum_t\eta_t=\infty\) and \(\sum_t\eta_t^2<\infty\). Neural-network objectives generally violate convexity, so these results are local tools and baselines rather than descriptions of the entire loss surface.

### 2.6 · CV2 — Computer Vision Classification and Augmentation

Image classification requires more than a final softmax. Normalize consistently, choose augmentations that preserve the label, inspect confusion matrices, and look at the examples the model gets wrong. Random crops, flips, color changes, mixup, and cutmix inject assumptions about which variations should be ignored, but an augmentation can erase the very signal the label depends on. Transfer learning usually starts from a pretrained backbone, freezes it for a first sanity check, then unfreezes with smaller learning rates so useful features are not destroyed. Self-supervised detours make the representation without labels: masked autoencoders reconstruct missing patches, while DINO-style teacher–student objectives encourage matching views without direct pixel reconstruction. Images tolerate larger masks than text because spatial redundancy lets context fill gaps. Linear probing and fine-tuning answer different questions; a good probe says the information is present, not that the representation is universally useful.

### 2.7 · CV3 — Semantic Segmentation with U-Net and Masks

Classification says what is in an image; segmentation says which pixels belong to it. An encoder can recognize a leaf while downsampling away the exact boundary, so U-Net sends skip connections from encoder stages into the decoder and restores fine spatial detail while retaining semantic context. Upsampling by interpolation is simple; transposed convolutions learn upsampling but can create checkerboard patterns when overlap is uneven. Pixelwise cross-entropy struggles when foreground is small, motivating overlap-sensitive losses such as Dice and IoU:

$$
\operatorname{Dice}=\frac{2|P\cap G|}{|P|+|G|},\qquad
\operatorname{IoU}=\frac{|P\cap G|}{|P\cup G|}.
$$

They carry nearly the same information because (\operatorname{Dice}=2\operatorname{IoU}/(1+\operatorname{IoU})), though their optimization gradients differ. Evaluate per class and inspect boundaries; a high average can hide a model that never finds rare objects. The encoder–decoder pattern is a recurring way to trade global abstraction for local reconstruction.

### 2.8 · CV4 — Modern Vision Architectures, ResNet, EfficientNet, and Transfer

A residual block computes \(y=x+F(x;\theta)\). Its input Jacobian is

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x},
$$

so backpropagation contains an identity term in addition to the derivative through \(F\). This does not guarantee nonvanishing gradients—the product of many \(I+J_F\) factors can still be poorly conditioned—but initialization with small residual branches keeps each factor near \(I\) and makes the network initially close to an identity map. A GNN self-loop is different because its self-message is generally transformed by the same weight matrix as neighbor messages.

Hierarchical vision backbones reduce spatial resolution as channel count grows. EfficientNet selects depth \(d=\alpha^\phi\), width \(w=\beta^\phi\), and resolution \(r=\gamma^\phi\) under an approximate compute constraint such as \(\alpha\beta^2\gamma^2\approx2\), instead of scaling one dimension alone. Window attention reduces a global \(O(N^2d)\) score computation to \(O(NM^2d)\) for windows containing \(M^2\) tokens; shifted windows let tokens cross the previous window boundaries in later layers. Transfer learning then changes the optimization problem: freezing the backbone tests the existing features, while fine-tuning updates them and can overwrite useful structure if the steps are too large.

### 3.2 · CV5 — Object Detection with FPN and YOLO

Detection must answer both “what?” and “where?”, so the output is a set of boxes, classes, and confidence scores rather than one label. Intersection over Union measures box overlap, non-maximum suppression removes redundant boxes, and mean average precision summarizes precision–recall behaviour across classes and IoU thresholds. Anchor-based detectors match boxes to preset shapes; anchor-free methods predict centres, sizes, or distributions directly. A feature pyramid network combines high-resolution shallow features with semantically strong deep features, helping detect both small and large objects. YOLO-style systems emphasize a single fast prediction pipeline, while two-stage detectors spend more computation on proposals. Focal loss downweights easy negatives when the background vastly outnumbers objects. Detection is a structured-output problem: the locality prior of CNNs helps, but the global-context tradeoff leads naturally toward attention and ViT-style backbones.

## Arc IV — Transformers and attention: letting representations look things up

### 2.9 · TF1 — Transformer Fundamentals

Given input \(X\in\mathbb R^{n\times d_{\text{model}}}\), one attention head forms

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

The score from token \(i\) to token \(j\) is the bilinear form \(q_i^\top k_j\). Row-wise softmax converts these scores into nonnegative coefficients summing to one, and the output is the corresponding weighted sum of value vectors:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The scale \(1/\sqrt{d_k}\) follows from a variance calculation. If the coordinates of \(q\) and \(k\) are independent, have mean \(0\), and have variance \(1\), then

$$
\operatorname{Var}(q^\top k)
=\operatorname{Var}\!\left(\sum_{\ell=1}^{d_k}q_\ell k_\ell\right)
=\sum_{\ell=1}^{d_k}\operatorname{Var}(q_\ell k_\ell)
=d_k.
$$

Dividing by \(\sqrt{d_k}\) keeps the score variance near \(1\) as the head dimension grows. Without the scaling, typical score differences grow like \(\sqrt{d_k}\); softmax then puts nearly all mass on the largest score and its Jacobian \(\partial p_i/\partial s_j=p_i(\delta_{ij}-p_j)\) becomes small for most coordinates. The assumptions are an initialization heuristic, but they explain the factor.

Separate \(W_Q,W_K,W_V\) matrices are required because the similarity space and the vectors being averaged need not be the same representation. Multi-head attention performs this calculation with \(h\) separately parameterized projections, concatenates the \(h\) outputs, and applies \(W_O\). Heads are not guaranteed to learn clean human-readable roles; the concrete gain is that the layer can use several lower-dimensional bilinear score functions and several value projections in parallel. Without positional information, permuting the rows of \(X\) permutes the output rows in the same way, so attention alone cannot distinguish token order. A causal mask sets disallowed logits to \(-\infty\) before softmax. It permits parallel computation of all training positions because the target prefix is known, while generation remains sequential because the next prefix depends on the token just sampled.

### 2.10 · TF2 — BERT and Masked Language Modeling

BERT is an encoder-only transformer trained to reconstruct selected masked tokens using both left and right context. In the common corruption rule, about (80\%) of selected tokens become `[MASK]`, (10\%) become a random token, and (10\%) stay unchanged, forcing the model to avoid relying on one visible cue. The attention mask lets each position use both left and right context; this is what makes the representation bidirectional. Pretraining learns a general contextual representation; fine-tuning adds a task head or adapts the whole model. Masked prediction has high signal density because each example supplies many surrounding tokens, but the objective alone does not guarantee every desired capability. Tokenization, sentence boundaries, and negative examples still matter. BERT’s architecture is a response to tasks where we want to understand a whole sequence at once, rather than produce the next token online.

### 2.11 · TF3 — GPT and Causal Language Modeling

GPT chooses a decoder-only architecture and trains on the exact chain-rule factorization

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

The causal mask makes each position predict only its future target, so all positions can train in parallel even though generation is sequential. Perplexity is the exponentiated average negative log-likelihood; old (n)-gram models used counts and smoothing, while GPT learns the conditional distribution with attention. Greedy decoding takes the most likely next token, beam search keeps several candidates, temperature changes concentration, top-(k) caps choices, and top-(p) keeps a probability nucleus. Sampling and beam search make different tradeoffs for open-ended prose. Evaluation should include factuality, repetition, calibration, and task success because likelihood and human preference are not identical objectives.

### 2.12 · TF4 — T5 and Encoder–Decoder Models

T5 treats text-to-text tasks as a shared interface: an encoder reads the input, a decoder generates the output, and decoder queries use cross-attention to the encoder’s keys and values. This avoids forcing every task into one classification head and is natural for translation, summarization, and structured generation. Teacher forcing supplies the true previous target during training, making optimization easier but creating exposure bias because inference must condition on its own earlier mistakes. The encoder can process its input bidirectionally while the decoder remains causal. Compared with compressing the entire source into one fixed vector, cross-attention lets each output token retrieve a different relevant part of the source. The same mechanism later connects image features to text. The architectural choice follows the direction of information flow: understand one stream, then produce another while repeatedly looking back.

### 3.3 · CV6 — Vision Transformers

A Vision Transformer divides an \(H\times W\) image into \(P\times P\) patches, producing \(N=(H/P)(W/P)\) tokens after a learned linear projection of each flattened patch. One attention layer connects every pair of patches, whereas a stack of local \(k\times k\) convolutions needs depth for distant pixels to share a receptive field. This changes the inductive bias: convolution imposes locality and translation-equivariant parameter sharing, while a ViT must learn useful spatial relations from data together with its positional encoding. The attention score matrix has \(N^2\) entries and costs \(O(N^2d)\) arithmetic at width \(d\). Halving \(P\) in both image dimensions multiplies \(N\) by four and the quadratic attention term by sixteen. Higher-resolution fine-tuning must also define positional information on the new patch grid, commonly by interpolation or a relative-position scheme. Pretraining can compensate for the weaker built-in spatial assumptions, but the result depends on dataset size, augmentation, resolution, and compute.

### 3.4 · NL1 — Tokenization and Embeddings

Text must become discrete tokens before a transformer can process it, and the tokenization scheme changes the problem. Character tokens handle any spelling but create long sequences; word tokens are compact but have an enormous vocabulary and fail on rare words. Subword methods such as BPE repeatedly merge frequent symbol pairs, giving common words short encodings while decomposing unfamiliar words. Leading spaces, punctuation, numbers, and Unicode can produce surprising token boundaries, so inspect the actual token IDs instead of trusting visual similarity. Embeddings begin with the distributional hypothesis: words appearing in similar contexts acquire nearby vectors. Word2Vec skip-gram predicts context from a centre word, often using negative sampling; GloVe uses a related count-based geometry. Analogy tests may reveal some structure in embeddings, but they do not establish clean semantic axes. Tokenization is lossy compression with a cost paid in sequence length, vocabulary size, multilingual coverage, and how easily the model can manipulate numbers.

### 3.5 · NL2 — Sequence Models and Transfer Learning

An RNN updates a hidden state sequentially, so memory is cheap but time parallelism is poor. Backpropagation through time multiplies Jacobians across steps; eigenvalues below one vanish and above one explode, though real gates and nonlinearities make the behaviour less perfectly symmetric. An LSTM adds input, forget, and output gates plus a cell state that creates a gradient highway; initializing the forget gate positively encourages remembering early. GRUs use fewer gates and parameters but make a different capacity–simplicity trade. Encoder–decoder RNNs expose a fixed-vector bottleneck, and Bahdanau additive attention fixes it by learning alignment weights over encoder states. Transfer learning then moves from pretraining objectives to fine-tuning, prompting, in-context learning, instruction tuning, and preference optimization. Chain-of-thought text is a behaviour elicited from a model, not automatically a faithful window into the computation that produced the answer.

### 3.6 · TF5 — RAG and Retrieval Embeddings

If a model’s parametric memory is stale, give it a small library to look things up in. Retrieval-augmented generation separates retrieve from generate: chunk documents, embed them, search with sparse lexical features, dense vectors, or a hybrid, rerank a shortlist, and pass evidence to the generator. Approximate nearest-neighbour indexes trade a little exactness for speed. Chunk size, overlap, metadata, and query rewriting often matter more than changing the embedding model. Measure retrieval recall and precision separately from answer faithfulness; a fluent unsupported answer is still wrong. Evaluate on questions whose evidence and time boundary are known, just as a train/test split protects classical ML. Failure modes include retrieving a plausible but irrelevant chunk, splitting a definition from its qualification, exceeding the context budget, and asking the generator to reconcile contradictory sources. RAG is not extra knowledge magically implanted in weights; it is a pipeline that gives a model conditional access to evidence.

### 3.7 · TF6 — Relative Position and RoPE

Self-attention sees a set unless positions change the representation. Absolute learned embeddings are simple but tied to the training range; sinusoidal features allow a formula at new positions but still encode absolute phase. Rotary position embeddings rotate query and key pairs by position-dependent angles. In two dimensions,

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix},
$$

and rotating (q) by one angle and (k) by another makes their dot product depend on the angle difference, hence relative offset. Extrapolating far beyond training can still fail when the model encounters frequencies and attention patterns outside those seen during training. Scaling tricks alter the effective frequency range, but must be evaluated on retrieval at multiple depths, exact position-sensitive tasks, and ordinary perplexity. Position is a modelling decision about which relationships should be easy to express.

### 3.8 · TF7 — Cross-Attention and Multimodal Attention

Cross-attention is the point where two streams actually meet: queries come from one sequence and keys and values from another. In image captioning, text queries retrieve image features; in translation, target tokens retrieve source states. A dual encoder embeds each modality independently and is fast for retrieval, a cross-encoder jointly attends and is more expressive but expensive, and connector modules such as learned query tokens compress a large visual feature map for a language model. CLIP aligns image and text embeddings with a contrastive objective, which enables retrieval and zero-shot classification but does not mean CLIP can explain every object or generate text. Vision-language systems must solve alignment, fusion, and generation separately. Resolution creates a token-budget problem, and shortcuts create modality collapse or object hallucination. A blank-image diagnostic can reveal whether a captioner is reading the image or merely producing a strong language prior.

### 4.2 · TF8 — GQA and MLA Attention Complexity

During autoregressive decoding, keys and values from earlier tokens are cached. Multi-head attention stores separate K/V heads, multi-query attention shares one K/V head across all query heads, and grouped-query attention shares K/V within groups, shrinking the cache while preserving more query diversity than MQA. Multi-head latent attention compresses the cached information into a latent representation and reconstructs the pieces needed by queries. The trade is bandwidth, memory, and representational freedom. Prefill processes the prompt in parallel; decode adds one token at a time and is often memory-bandwidth bound. PagedAttention stores cache blocks so variable-length requests do not waste contiguous memory. These changes leave the attention definition intact while changing its memory and compute costs. Measure end-to-end latency, cache size, and quality; a smaller theoretical cache may still be slower if reconstruction or the kernel becomes awkward.

### 4.3 · TF9 — Efficient Attention and Long Context

There are two different ways to make attention cheaper. FlashAttention keeps the exact softmax result but tiles the computation so the (n\times n) matrix need not be written to memory; online softmax tracks row maxima and normalizers while blocks arrive. Approximate methods change the algorithm: sparse patterns restrict pairs, linear attention reassociates feature maps,

$$
\operatorname{softmax}(QK^\top)V\ \approx\ \phi(Q)\big(\phi(K)^\top V\big),
$$

and Performer, Linformer, Nyström, and hashing each make a different low-rank or sampling assumption. Sliding windows save cost but restrict long jumps; state-space models and Mamba compress history into a recurrent state, gaining speed while making precise retrieval harder. Long-context evaluation needs needle-in-a-haystack probes, induction-head tests, position-depth sweeps, and cache-memory measurements, not perplexity alone. Attention sinks and cache eviction are practical responses to finite memory. The exact baseline has improved so much that every approximate “win” needs a fair comparison against tiled exact attention.

## Arc V — Representation, generative, and advanced topics

### 2.13 · RP1 — Representation Learning: Autoencoders, VAEs, and Latent Spaces

An autoencoder chooses an encoder \(z=f_\phi(x)\in\mathbb R^k\) and decoder \(\hat x=g_\theta(z)\), then minimizes a reconstruction loss. With centered data, linear maps, squared error, and a rank-\(k\) bottleneck, the composite decoder–encoder matrix has rank at most \(k\). The Eckart–Young theorem therefore makes the optimum reconstruction \(X_k=U_k\Sigma_kV_k^\top\), so the learned code spans the PCA principal subspace, although its coordinates can differ by an invertible change of basis. Nonlinear autoencoders are not constrained to PCA and can memorize unless the bottleneck or another regularizer actually limits them.

After training, \(f_\phi(x)\) can be used as a feature vector. Reconstruction error \(r(x)=\lVert x-g_\theta(f_\phi(x))\rVert^2\) can also be an anomaly score, but it is not a probability. A threshold can be estimated from held-out in-distribution errors, for example by an empirical upper quantile, and then evaluated against the desired false-positive rate under distribution shift. Generation exposes a separate issue: the aggregate set of encodings \(f_\phi(x_i)\) need not fill \(\mathbb R^k\), so a randomly chosen latent point can lie in a region where the decoder received no training constraint.

### 2.14 · RP2 — Contrastive Learning

Let \(z_i\) and \(z_j\) be normalized embeddings of two augmented views of the same example. InfoNCE treats \(j\) as the correct class among candidate matches:

$$
\mathcal L_i=-\log\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
$$

If \(s_k=\operatorname{sim}(z_i,z_k)\) and \(p_k=\operatorname{softmax}(s_k/\tau)\), then

$$
\frac{\partial\mathcal L_i}{\partial s_k}
=\frac{p_k-\mathbf 1[k=j]}{\tau}.
$$

Smaller \(\tau\) makes the softmax more concentrated and increases the gradient scale, so it changes both which negatives dominate and the optimization dynamics. SimCLR obtains negatives from a large batch; MoCo uses a queue whose embeddings come from a slowly updated encoder. The augmentation distribution defines which transformations form positive pairs. If a crop removes the class-defining feature, the objective still forces the two views together and teaches an unwanted invariance.

Methods without explicit negatives need another mechanism to avoid the constant solution \(z_i=c\). BYOL and SimSiam use stop-gradient plus architectural asymmetry, Barlow Twins penalizes cross-correlation away from the identity, and VICReg includes explicit variance and covariance terms. A linear probe measures linear accessibility of a chosen label under a fixed dataset and optimizer; it does not characterize every property of the representation.

### 3.1 · PS4 — Information Theory

For a discrete distribution \(p\), entropy is \(H(p)=-\sum_xp(x)\log p(x)\). If samples come from \(p\) but are scored with \(q\), the expected negative log score is the cross-entropy

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_xp(x)\log\frac{p(x)}{q(x)},\qquad H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q).
$$

Minimizing empirical negative log-likelihood is therefore minimizing an empirical estimate of \(H(p,q)\); the unknown \(H(p)\) does not depend on the model, so the population optimum also minimizes \(D_{\mathrm{KL}}(p\Vert q)\).

The support and weighting in the expectation explain the difference between KL directions more accurately than the labels “mode-covering” and “mode-seeking.” In

$$
D_{\mathrm{KL}}(p\Vert q)=\mathbb E_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right],
$$

any region with \(p(x)>0\) and \(q(x)=0\) gives infinite divergence, while assigning extra \(q\)-mass where \(p\) is tiny is penalized only indirectly through normalization. In

$$
D_{\mathrm{KL}}(q\Vert p)=\mathbb E_{x\sim q}\!\left[\log\frac{q(x)}{p(x)}\right],
$$

regions that \(q\) does not visit contribute nothing. When a restricted unimodal \(q\) approximates a separated multimodal \(p\), placing mass between modes can incur a large penalty because \(p\) is small there, so one mode may minimize the objective. These are consequences of the expectation measure and the model family, not universal behavioral definitions of the two expressions.

Mutual information is itself a KL divergence,

$$
I(X;Y)=D_{\mathrm{KL}}(p_{XY}\Vert p_Xp_Y)
=H(X)-H(X\mid Y).
$$

For a Markov chain \(X\to Y\to Z\), the chain rule gives \(I(X;Y)=I(X;Z)+I(X;Y\mid Z)\), hence \(I(X;Z)\le I(X;Y)\). This is the data-processing inequality. Information-bottleneck objectives and variational mutual-information bounds are particular optimization constructions built from these identities; they do not imply that every trained network follows one monotone compression trajectory.

### 3.9 · RP4 — Autoregressive Generative Models

Autoregressive generation begins with the chain rule, not a particular architecture:

$$
p(x_{1:T})=\prod_t p(x_t\mid x_{<t}).
$$

The factorization is exact for any chosen ordering, but sampling requires one conditional evaluation per dimension. That is fine for a thousand text tokens and painful for (3\times256\times256=196{,}608) pixels. Teacher-forced maximum likelihood is stable because each conditional is a supervised prediction, and perplexity summarizes held-out token likelihood. Greedy, beam, temperature, top-(k), and top-(p) decoding choose different quality–diversity compromises; beam search is useful for constrained translation and often dull for open-ended text. PixelCNN’s raster order is also an image-specific inductive bias, since a pixel cannot see future raster positions. VQ-VAE compresses an image into a shorter grid of discrete codes so a sequence model can operate over perhaps (32^2) rather than (256^2) positions. Exact likelihood is valuable, but sequential sampling is a real systems cost.

### 3.10 · RP3 — Variational Autoencoders and Generation

A VAE specifies a latent-variable model \(p_\theta(x,z)=p(z)p_\theta(x\mid z)\) and an encoder \(q_\phi(z\mid x)\) that approximates the posterior. Insert \(q_\phi\) into the marginal log likelihood:

$$
\log p_\theta(x)
=\mathbb E_{q_\phi(z\mid x)}
\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\right).
$$

The KL term is nonnegative, so the first expectation is a lower bound. Expanding \(p_\theta(x,z)=p_\theta(x\mid z)p(z)\) gives the usual ELBO:

$$
\log p_\theta(x)\ge\mathcal L(\theta,\phi;x)
=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p(z)\right).
$$

For (q=\mathcal N(\mu,\operatorname{diag}(\sigma^2))) and (p=\mathcal N(0,I)),

$$
D_{\mathrm{KL}}(q\Vert p)=\frac12\sum_j\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right).
$$

For a diagonal Gaussian encoder and unit-normal prior, the KL has a closed form because the log-density ratio separates across dimensions. The reconstruction term is an expected log likelihood; MSE corresponds to a Gaussian decoder with fixed variance, while binary cross-entropy corresponds to independent Bernoulli output coordinates. To differentiate its Monte Carlo estimate, write

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
$$

with \(\epsilon\) sampled independently of \(\phi\). The sample is then a differentiable function of \(\mu_\phi(x)\) and \(\sigma_\phi(x)\). Implementations predict \(\ell=\log\sigma^2\) because \(\ell\in\mathbb R\) is unconstrained and \(\sigma=\exp(\ell/2)>0\); this also avoids optimizing a variance directly against a positivity boundary. A diagonal covariance removes \(O(d^2)\) covariance parameters and the corresponding matrix factorization, but it cannot represent posterior correlations. If \(p_\theta(x\mid z)\) can model the data while ignoring \(z\), maximizing the ELBO may set \(q_\phi(z\mid x)=p(z)\), making the KL zero; this is posterior collapse. KL annealing and free bits alter the optimization pressure but do not change why the collapse is a valid optimum.

### 3.11 · GM1 — Generative Adversarial Networks

A GAN has a generator (G) producing samples and a discriminator (D) separating real from fake. The minimax game is

$$
\min_G\max_D\ \mathbb E_{x\sim p_{data}}\log D(x)+\mathbb E_{z\sim p(z)}\log(1-D(G(z))).
$$

For fixed generator density \(p_g\), the discriminator objective is pointwise:

$$
\int\left[p_{\mathrm{data}}(x)\log D(x)+p_g(x)\log(1-D(x))\right]dx.
$$

Maximizing \(a\log D+b\log(1-D)\) gives \(D^*(x)=a/(a+b)\), hence

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}.
$$

Substitution yields \(-\log4+2D_{\mathrm{JS}}(p_{\mathrm{data}}\Vert p_g)\). This calculation assumes an optimal discriminator and densities on a common support; it does not describe the gradients available during alternating finite-step training. With the saturating generator objective \(\mathbb E_z\log(1-D(G(z)))\), a confident discriminator can make the sigmoid derivative small. The non-saturating loss \(-\mathbb E_z\log D(G(z))\) has the same desired fixed point but stronger gradients early in training. Mode collapse means \(G_\#p(z)\) places too much probability on a subset of the data support. Visual fidelity alone cannot detect that loss of coverage, which is why evaluation needs distributional or recall-like measurements in addition to sample quality.

### 3.12 · GM2 — Diffusion Models

Choose a forward Markov process

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
$$

where \(\bar\alpha_t=\prod_{s=1}^t\alpha_s\). This closed form follows by repeatedly substituting \(x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\) and combining independent Gaussian noise terms. For a schedule with \(\bar\alpha_T\approx0\), \(x_T\) is close to \(\mathcal N(0,I)\). Conditioned on \(x_0\), the posterior \(q(x_{t-1}\mid x_t,x_0)\) is Gaussian because it is obtained by conditioning a joint Gaussian. A model \(p_\theta(x_{t-1}\mid x_t)\) approximates this reverse transition.

With the common fixed-variance parameterization, the variational objective reduces, up to timestep-dependent weights, to

$$
\mathbb E_{t,x_0,\epsilon}\left[
\left\lVert\epsilon-\epsilon_\theta(
\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)
\right\rVert_2^2\right].
$$

Predicting \(\epsilon\) is equivalent to estimating the conditional mean needed by the reverse step; it is also related to the score \(\nabla_{x_t}\log q_t(x_t)\). DDPM sampling evaluates a reverse transition at every selected timestep. DDIM, higher-order solvers, and distillation reduce the number of evaluations by changing the reverse trajectory or learning to skip steps. Classifier-free guidance combines conditional and unconditional noise estimates, usually \(\epsilon_{\mathrm{cfg}}=(1+w)\epsilon_\theta(x_t,c)-w\epsilon_\theta(x_t,\varnothing)\); larger \(w\) strengthens conditioning but changes the sampled distribution and can reduce diversity.

### 3.13 · XT1 — Physics-Informed Neural Networks

A physics-informed neural network combines fitting observed values with a PDE residual. It predicts a field (u_\theta(x,t)), uses automatic differentiation to compute derivatives, and adds the residual to the loss:

$$
\mathcal L=\lambda_{data}\mathcal L_{data}+\lambda_{PDE}\lVert\mathcal F[u_\theta](x,t)\rVert^2
+\lambda_{bc}\mathcal L_{boundary}+\lambda_{ic}\mathcal L_{initial}.
$$

The relative loss weights matter because multiplying one term by a constant multiplies its gradient by the same constant. A numerically dominant boundary loss can therefore suppress progress on the interior residual even when both terms are conceptually important. Spectral bias can also make high-frequency solutions hard to represent. Compare against a converged classical solver and evaluate residuals between collocation points, on held-out parameters, and under mesh refinement. Neural operators change the target from one solution \(u\) to an operator mapping an input function or coefficient field to \(u\); DeepONet and Fourier Neural Operator impose different parameterizations of that map. Automatic differentiation evaluates derivatives of \(u_\theta\), but it does not verify that the selected PDE, boundary conditions, or collocation distribution describe the physical system.

### 4.1 · PS5 — Applied Information Theory

Applied information theory asks what a training signal teaches a representation. Masked prediction, denoising, and contrastive objectives all create pretext tasks whose useful content depends on what the corruption leaves predictable. Mutual-information bounds can motivate an objective but are often loose and estimator-dependent. Compression has a hard limit: if the representation discards a variable required by the task, no classifier head can recover it. A frozen linear probe tests accessibility, fine-tuning tests adaptability, and transfer across tasks tests whether the representation captured reusable structure; disagreement among these probes is informative. Bits-per-dimension evaluates likelihood models, while GANs need other metrics. The information bottleneck language is a way to reason about the tension between retaining signal and discarding nuisance variation. State which distribution, decoder, or probe the measurement depends on before interpreting it.

### 4.4 · RP5 — Vision–Language Models

Vision–language systems have at least three problems: align image and text representations, fuse them so one can condition on the other, and generate a useful output. Dual encoders such as CLIP are efficient for retrieval and zero-shot matching; cross-encoders are richer but require joint computation; connector modules feed a frozen or lightly tuned vision tower into a language model, sometimes through learned query tokens. Contrastive data quality matters more than a perfect loss if captions are noisy or object coverage is poor. Image-conditioned generation often uses latent diffusion with cross-attention, while high-resolution inputs force a choice between more visual tokens and learned compression. Modality collapse occurs when one stream dominates, and hallucination occurs when the language prior fills in an object the image never supported. Embedding similarity is not object verification. Test blank or corrupted images, counterfactual captions, fine-grained retrieval, and generation faithfulness separately.

### 4.5 · RF1 — Reinforcement Learning

An MDP specifies states, actions, transition probabilities \(P(s'\mid s,a)\), rewards, and a discount \(0\le\gamma<1\). For a policy \(\pi\), the value is the expected discounted return

$$
V^\pi(s)=\mathbb E_\pi\!\left[\sum_{k=0}^\infty\gamma^kr_{t+k+1}\mid s_t=s\right].
$$

The Bellman optimality operator is

$$
(TV)(s)=\max_a\mathbb E[r+\gamma V(s')\mid s,a].
$$

For any bounded \(V,W\), the inequality \(|\max_af_a-\max_ag_a|\le\max_a|f_a-g_a|\) gives

$$
\lVert TV-TW\rVert_\infty
\le\gamma\lVert V-W\rVert_\infty.
$$

Thus \(T\) is a contraction. Banach’s fixed-point theorem gives a unique \(V^*\) and convergence of value iteration \(V_{k+1}=TV_k\) at rate at most \(\gamma^k\). Monte Carlo estimates returns without bootstrapping; temporal-difference methods replace the unknown continuation return by a current value estimate. SARSA targets \(r+\gamma Q(s',a')\) with \(a'\) drawn from the behavior policy, while Q-learning targets \(r+\gamma\max_{a'}Q(s',a')\), making it off-policy.

For a differentiable stochastic policy, the likelihood-ratio identity yields

$$
\nabla_\theta J(\theta)
=\mathbb E_{\tau\sim\pi_\theta}\!\left[
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t
\right].
$$

Subtracting a baseline depending only on \(s_t\) leaves the expectation unchanged because \(\mathbb E_{a\sim\pi}[\nabla_\theta\log\pi(a\mid s)]=0\), while it can reduce variance. Actor–critic estimates this baseline with a value function. Replay buffers and target networks in DQN address correlation and moving targets, but combining function approximation, bootstrapping, and off-policy data can still diverge. PPO and preference-based methods add further approximations, so their guarantees are not inherited from the tabular contraction argument.

### 4.6 · XT2 — Graph Neural Networks

Not every dataset is a sequence or a grid. Imagine nodes are people and edges are relationships likely to carry information. A message-passing layer gathers neighbour representations, transforms them, includes a self-message, applies a nonlinearity, and repeats. With self-loops (\tilde A=A+I), a normalized GCN is

$$
H^{(l+1)}=\sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\right).
$$

For a tiny graph with

$$
A=\begin{bmatrix}0&1&1\\1&0&0\\1&0&0\end{bmatrix},\quad
H=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},
$$

the first row of \(AH\) is the sum of nodes 2 and 3, \((1,2)\). With unnormalized aggregation, node \(i\) receives \(\sum_{j\in N(i)}h_j\). If neighbor features are independent with common mean \(\mu\) and covariance \(\Sigma\), this sum has mean \(d_i\mu\) and covariance \(d_i\Sigma\), so both its scale and variance depend on degree. Left normalization \(D^{-1}A\) computes a mean, removing the mean’s degree factor but making the operator asymmetric even for an undirected graph. Symmetric normalization \(S=D^{-1/2}AD^{-1/2}\) weights an edge \((i,j)\) by \(1/\sqrt{d_id_j}\). For an undirected graph, \(S\) is symmetric and similar to the random-walk matrix:

$$
D^{-1}A=D^{-1/2}SD^{1/2}.
$$

It therefore has the same eigenvalues while remaining self-adjoint in the ordinary Euclidean inner product. Adding self-loops before computing degrees gives \(\tilde S=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}\); the self contribution is still multiplied by \(W^{(l)}\), so this is not an identity residual path.

Graph attention replaces fixed normalized weights with learned coefficients. A standard GAT computes

$$
e_{ij}=\operatorname{LeakyReLU}\!\left(a^\top[Wh_i\Vert Wh_j]\right),\qquad
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k\in N(i)}\exp(e_{ik})},
$$

and updates \(h_i'=\sigma(\sum_{j\in N(i)}\alpha_{ij}Wh_j)\). GraphSAGE instead samples neighbors and uses a fixed mean, sum, or max aggregator; GIN uses a sum followed by an MLP. Stacking \(L\) message-passing layers limits direct dependence to the \(L\)-hop neighborhood. Repeated multiplication by a normalized propagation matrix attenuates components outside its leading eigenspaces, which explains oversmoothing. Oversquashing is different: exponentially many distant signals may have to pass through a small graph cut and be represented in a fixed-width state.

## Coverage ledger

| Source module ID | Heading in this edition |
| --- | --- |
| 1.1 PY1 | NumPy Programming |
| 1.2 PY2 | Pandas Data Cleaning |
| 1.3 PY3 | Data Visualization |
| 1.4 PY4 | Colab and Markdown Notebooks |
| 1.5 PY5 | Scikit-Learn Pipelines |
| 1.6 PY6 | PyTorch Fundamentals |
| 1.7 MA1 | Vectors, Matrices, Systems, Transformations |
| 1.8 MA2 | Eigenvalues and Spectral Theory |
| 1.9 MA3 | Calculus and Gradients |
| 1.10 PS1 | Probability and Bayes |
| 1.11 PS2 | Expectation and Joint Distributions |
| 1.12 PS3 | Statistical Inference and Hypothesis Testing |
| 1.13 ML1 | Data Processing and Features |
| 1.14 ML2 | Supervised Learning Foundations |
| 1.15 ML3 | Kernel Methods and SVMs |
| 1.16 ML4 | Tree-Based Models and Ensembles |
| 1.17 ML5 | Clustering |
| 1.18 DL1 | Neural Networks, Perceptrons, MLPs, Activations |
| 1.19 DL2 | Regularization and BatchNorm |
| 1.20 DL3 | Model Training and Generalization |
| 1.21 DL4 | Optimization: Adam and SGD |
| 1.22 CV1 | Convolutional Neural Networks |
| 2.1 MA4 | Backpropagation Mechanics |
| 2.2 MA5 | Advanced Linear Algebra: Projections, SVD, Low Rank |
| 2.3 ML6 | Model Evaluation |
| 2.4 ML7 | Dimensionality Reduction |
| 2.5 DL5 | Advanced Optimization |
| 2.6 CV2 | Computer Vision Classification and Augmentation |
| 2.7 CV3 | Semantic Segmentation with U-Net and Masks |
| 2.8 CV4 | Modern Vision Architectures, ResNet, EfficientNet, and Transfer |
| 2.9 TF1 | Transformer Fundamentals |
| 2.10 TF2 | BERT and Masked Language Modeling |
| 2.11 TF3 | GPT and Causal Language Modeling |
| 2.12 TF4 | T5 and Encoder–Decoder Models |
| 2.13 RP1 | Representation Learning: Autoencoders, VAEs, and Latent Spaces |
| 2.14 RP2 | Contrastive Learning |
| 3.1 PS4 | Information Theory |
| 3.2 CV5 | Object Detection with FPN and YOLO |
| 3.3 CV6 | Vision Transformers |
| 3.4 NL1 | Tokenization and Embeddings |
| 3.5 NL2 | Sequence Models and Transfer Learning |
| 3.6 TF5 | RAG and Retrieval Embeddings |
| 3.7 TF6 | Relative Position and RoPE |
| 3.8 TF7 | Cross-Attention and Multimodal Attention |
| 3.9 RP4 | Autoregressive Generative Models |
| 3.10 RP3 | Variational Autoencoders and Generation |
| 3.11 GM1 | Generative Adversarial Networks |
| 3.12 GM2 | Diffusion Models |
| 3.13 XT1 | Physics-Informed Neural Networks |
| 4.1 PS5 | Applied Information Theory |
| 4.2 TF8 | GQA and MLA Attention Complexity |
| 4.3 TF9 | Efficient Attention and Long Context |
| 4.4 RP5 | Vision–Language Models |
| 4.5 RF1 | Reinforcement Learning |
| 4.6 XT2 | Graph Neural Networks |
