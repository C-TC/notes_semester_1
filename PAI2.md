
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Bayesian Deep Learning](#bayesian-deep-learning)
  - [Lecture Notes](#lecture-notes)
    - [Variational Inference for Bayesian neural networks](#variational-inference-for-bayesian-neural-networks)
    - [Markov-Chain Monte Carlo for Bayesian Neural Networks](#markov-chain-monte-carlo-for-bayesian-neural-networks)
    - [Specialized Inference Techniques for Bayesian Neural Networks](#specialized-inference-techniques-for-bayesian-neural-networks)
- [Active Learning](#active-learning)
  - [Lecture Notes](#lecture-notes-1)
- [Bayesian Optimization](#bayesian-optimization)
  - [Lecture Notes](#lecture-notes-2)
- [Markov Decision Processes](#markov-decision-processes)
  - [Lecture Notes](#lecture-notes-3)

<!-- /code_chunk_output -->

## Bayesian Deep Learning
### Lecture Notes
**Bayesian Learning**
* Prior: $p(\theta)$
* Likelihood: $p(y_{1:n}|x_{1:n},\theta)\prod_{i=1}^n p(y_i|x_i,\theta)$
* Posterior: $p(\theta|x_{1:n},y_{1:n})=\frac{1}{Z}p(\theta)\prod_{i=1}^n p(y_i|x_i,\theta)$
where $Z=\int p(\theta)\prod_{i=1}^np(y_i|x_i,\theta)d\theta$
* Predictions: $p(y^*|x^*,x_{1:n},y_{1:n})=\int p(y^*|x^*,\theta)p(\theta|x_{1:n},y_{1:n})d\theta$

**Beyond Linear Models**
* So far, we’ve discussed effective approximate inference techniques for **Bayesian linear regression** and **Bayesian logistic regression** (linear  classification)
$p(y|\mathbf{x},\theta)=\mathcal{N}(y;\theta^T\mathbf{x},\sigma^2)$
$p(y|\mathbf{x},\theta)=Ber(y;\sigma(\theta^T\mathbf{x}))$
* Here, likelihood functions have parameters **linearly dependent** on the inputs
* In practice, can often get better performance by considering **nonlinear** dependencies

**(Deep) Artificial Neural Networks**
* $f(\mathbf{x};\mathbf{w})=\varphi(\mathbf{W}_1\varphi(\mathbf{W}_2(...\varphi(\mathbf{W}_l\mathbf{x}))))$
* Flexible nonlinear functions with many (often $~10^8$) parameters
* Deep = ''nested'' in many layers
* Loosely inspired by biological neuronal networks

**Some Deep Learning Success Stories**
* State of the art performance on some difficult classification tasks
* Speech recognition (TIMIT)
* Image recognition (MNIST, ImageNet)
* Natural language processing (semantic word embeddings)
* Speech translation
* A lot of recent work on sequential models (Recurrent Neural Networks, LSTMs, GRUs, Tramsformers,...) and models on graphs

**Activation Functions**
* Hereby, $\theta\in\mathbb{R}^d$ and $\varphi:\mathbb{R}\to \mathbb{R}$ is a nonlinear function, called ''activation function''
$\phi(\mathbf{x},\theta)=\varphi(\theta^T\mathbf{x})$

**Bayesian Neural Networks**
* Bayesian neural network models specify a prior distribution over weights, and use likelihood functions parameterized via neural networks
* Simple example:
    * Gaussian prior on weights: $p(\theta)=\mathcal{N}(\theta;0,\sigma_p^2I)$
    * Likelihood: $p(y|\mathbf{x},\theta)=\mathcal{N}(y;f(\mathbf{x},\theta),\sigma^2) \quad$ as opposed to $\theta^T\mathbf{x}$ in BLR

**Heteroscedastic Noise**
* Noise depends on input

**Modeling heteroscedastic noise with NNs**
* Use more complex likelihood:
    * $p(y|\mathbf{x},\theta)=\mathcal{N}(y;f_1(\mathbf{x},\theta),exp(f_2(\mathbf{x},\theta))) \quad$
    Model both mean and (log) variance as (two different) outputs of a neural network

**MAP Estimates for Bayesian NNs**
* MAP estimate for heteroscedastic regression with NNs
$\hat{\theta}=argmin_\theta -logp(\theta)-\sum_{i=1}^n logp(y_i|\mathbf{x}_i,\theta)$
$\begin{aligned}
logp(y|\mathbf{x},\theta)&=log\mathcal{N}(y;\mu(\mathbf{x},\theta),\sigma^2(\mathbf{x},\theta))\\
&=log(\frac{1}{\sqrt{2\pi\sigma^2(\mathbf{x},\theta)}}exp(-\frac{(y-\mu(\mathbf{x},\theta))^2}{2\sigma^2(\mathbf{x},\theta)}))\\
&=log\frac{1}{\sqrt{w\pi}}-\frac{1}{2}log\sigma^2(\mathbf{x},\theta)-\frac{1}{2}\frac{(y-\mu(\mathbf{x},\theta))^2}{\sigma^2(\mathbf{x},\theta)}
\end{aligned}$
* MAP estimate for heteroscedastic regression with NNs
$\begin{aligned}
\hat{\theta}&=argmin_\theta -logp(\theta)-\sum_{i=1}^n logp(y_i|\mathbf{x}_i,\theta)\\
&=argmin_\theta \lambda\|\theta\|_2^2+\sum_{i=1}^n[\frac{1}{2\sigma^2(\mathbf{x}_i,\theta)}\|y_i-\mu(\mathbf{x},\theta)\|^2+\frac{1}{2}log\sigma^2(\mathbf{x}_i,\theta)]
\end{aligned}$
* Thus, the network can **attenuate the losses** for certain data points by **attributing the error** to large variance

**Recall: MAP Inference in BNNs**
* Finding the **MAP parameter** estimates in BNNs can be accomplished by minimizing
$\hat{\theta}=argmin_\theta -logp(\theta)-\sum_{i=1}^n logp(y_i|\mathbf{x}_i,\theta)$
e.g., via Stochastic Gradient Descent (and variants)
* Gradients can be computed using auto-differentiation techiques (implemented, e.g., in PyTorch, Tensorflow)
* Gaussian priors on the weight are equivalent to
applying weight decay
$nabla (-logp(\theta))=\nabla \lambda\|\theta\|_2^2=2\lambda\theta$
$\theta_{t+1}\leftarrow \theta_t-\eta_t\nabla logp(\theta_t)-\eta_t\nabla\sum_{i=1}^nlogp(y_i|x_i,\theta_t)$
$\Rightarrow \theta_{t+1}\leftarrow \theta_t(1-2\lambda\eta_t)-\eta_t\nabla\sum_{i=1}^nlogp(y_i|x_i,\theta_t)$

**Approximate Inference for BNNs**
* Bayesian learning integrals for posterior and predictions for NN are intractable, thus need approximate inference.
* Can use approximate inference techniques at similar cost as MAP/SGD
    * Black-box stochastic variational inference
    * Stochastic gradient Langevin dynamics
* Only need to be able to **compute gradients**, which can be done using automatic differentiation (backpropagation)
* Also specialized approaches tailored for BNNs
    * Monte-carlo Dropout
    * Probabilistic Ensmbles

#### Variational Inference for Bayesian neural networks
**Variational inference for BNNs (aka Bayes by Backprop)**
* Recall, variational inference aims to find the best-approximating variational distribution $q$ via
$argmin_q KL(q\|p(\cdot|y))=argmax_q\mathbb{E}_{\theta\sim q(\theta)}[logp(y|\theta)]-KL(q\|p(\cdot))$
* For Gaussian $q(\theta|\lambda)=\mathcal{N}(\theta;\mu,\Sigma)$ can obtain stochastic gradients, e.g., via reparametrization trick
 $\begin{aligned}
\nabla_\lambda L(\lambda)&=\nabla_\lambda [\mathbb{E}_{\theta\sim q(\cdot|\lambda)}[logp(y|\theta)]-KL(q_\lambda\|p(\cdot))]\\
&=\nabla_{C,\mu}\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[logp(y|C\epsilon+\mu)]-\nabla_{C,\mu}KL(q_{C,\mu}\|p(\cdot))\\
&\approx  \nabla_{C,\mu}frac{n}{m} \sum_{j=1}^m logp(y_{i_j}|C\epsilon^{(j)}+\mu,x_{i_j})-\nabla_{C,\mu}KL(q_{C,\mu}\|p(\cdot))
\end{aligned}$

**Making Predictions**
* Given variational posterior q, can approximate predictive distributions by sampling from it
$\begin{aligned}
p(y^*|x^*,x_{1:n},y_{1:n})&=\int p(y^*|x^*,\theta)p(\theta|x_{1:n},y_{1:n})d\theta\\
&=\mathbb{E}_{\theta\sim p(\cdot|x_{1:n},y_{1:n})}[p(y^*|x^*,\theta)]\\
&\overset{V.I.}\approx \mathbb{E}_{\theta\sim q(\cdot|\lambda)}[p(y^*|x^*,\theta)]\\
&\overset{M.C.}\approx \frac{1}{m}\sum_{j=1}^m p(y^*|x^*,\theta^{(j)})
\end{aligned}$
where $\theta^{(j)}\approx q(\cdot|\lambda)$
* *Note*: one choice: $p(y^*|x^*,\theta^{(j)})=\mathcal{N}(y^*;\mu(x^*,\theta^{(j)}),\sigma^2(x^*,\theta^{(j)}))$
* i.e., **draw $m$ sets of weights** from posterior, and **average the neural network predictions**
* For Gaussian likelihoods, approximate predictive distribution becomes a mixture of Gaussians 

**Aleatoric vs. Epistemic Uncertainty for Gaussian Likelihoods**
* $p(y^*|\mathbf{X},\mathbf{y},\mathbf{x}^*)\approx \frac{1}{m}\sum_{j=1}^m \mathcal{N}(y^*;\mu(\mathbf{x}^*,\theta^{(j)}),\sigma^2(\mathbf{x}^*,\theta^{(j)}))$
* Mean: $\mathbb{E}[y^*|\mathbf{x}_{1:n},\mathbf{y}_{1:n},\mathbf{x}^*]\approx \bar{\mu}(\mathbf{x}^*):=\frac{1}{m}\sum_{m=1}^m \mu(\mathbf{x}^*,\theta^{(j)})$
* Law of Total Variance:
RVs. $\theta,y$, $Var(y)=\mathbb{E}_\theta Var[y|\theta]+Var\mathbb{E}_y[y|\theta]$
* Variance(via LoTV)
$\begin{aligned}Var[y^*|\mathbf{x}_{1:n},\mathbf{y}_{1:n},\mathbf{x}^*]&=\mathbb{E}[Var[y^*|\mathbf{x}^*,\theta]]+Var[\mathbb{E}[y^*|\mathbf{x}^*,\theta]]\\
&\approx \frac{1}{m}\sum_{j=1}^m \sigma^2(\mathbf{x}^*,\theta^{(j)})+\frac{1}{m}\sum_{j=1}^m (\mu(\mathbf{x}^*,\theta^{(j)}-\bar{\mu}(\mathbf{x}^*))^2
\end{aligned}$
where $\frac{1}{m}\sum_{j=1}^m \sigma^2(\mathbf{x}^*,\theta^{(j)})$ is Aleatoric uncertainty, and $\frac{1}{m}\sum_{j=1}^m (\mu(\mathbf{x}^*,\theta^{(j)}-\bar{\mu}(\mathbf{x}^*))^2$ is Epistemic uncertainty.

#### Markov-Chain Monte Carlo for Bayesian Neural Networks
**MCMC for Neural Networks**
* Similarly to variational inference, can apply **MCMC methods** to train deep neural network models such as
    * (Preconditioned) Stochastic Gradient Langevin Dynamics
    $\theta_{t+1}=\theta_t-\eta_t(\nabla logp(\theta_t)+\frac{n}{m}\sum_{j=1}^m \nabla logp(y_{i_j}|\theta_t,x_{i_j}))+\epsilon_t$
    * Metropolis adjusted Langevin Dynamics*
    * Stochastic Gradient Hamiltonian Monte Carlo
    * ...
* These methods **only require stochastic gradients** of the (unnormalized) joint probability, i.e., the same gradients used for MAP estimation (e.g., via SGD)

**Predicting with MCMC**
* MCMC methods (like SGLD) produce a sequence of iterates (NN weights) $\theta^{(1)},...,\theta^{(T)}$
* The ergodic theorem justifies making predictions with
$p(y^*|\mathbf{X},\mathbf{y},\mathbf{x}^*)\approx \frac{1}{T}\sum_{j=1}^T p(y^*|\mathbf{x}^*,\theta^{(j)})\approx \frac{1}{m}\sum_{j=1}^m p(y^*|\mathbf{x}^*,\theta^{(t_j)})\leftarrow j^{th}$snapshot
* Challenges:
    * Typically, **cannot afford to store** all T samples / models
    * To **avoid the “burn-in” period**, need to drop first samples

**Summarizing MCMC Iterates**
* Approach 1: **Subsampling**
    * Simply keep a subset of $m$ “snapshots
* Approach 2: **Gaussian approximation**
    * Keep track of a Gaussian approximation of the parameters $q(\theta|\mu_{1:d},\sigma^2_{1:d})$, where
    $\mu_i^{(T)}=\frac{1}{T}\sum_{i=1}^T\theta_i^{(j)}\qquad \sigma_i^2=\frac{1}{T}\sum_{j=1}^T(\theta_i^{(j)}-\mu_i)^2$
    $i$ is NN parameter index, $j$ is iteration of MCMC chain 
    * Can be implemented using running averages
    $\mu_i^{(t+1)}=\frac{1}{t+1}(t\mu_i^{(t)}+\theta_i^{(t+1)})$
    * To predict, sample weights from distribution $q$
    * Works well even when simply using SGD (no Gaussian noise) to generate $\theta^{(1)},...,\theta^{(T)}\Rightarrow$ **SWAG Method**

#### Specialized Inference Techniques for Bayesian Neural Networks
**Recall: Dropout Regularization**
* Key idea: randomly ignore (''drop out'') hidden units during each iteration of SGD with probability $p$

**Outlook: Dropout as Variational Inference**
* Dropout can be viewed as **performing variational inference*** with a particular variational family
$q(\theta|\lambda)=\prod_j q_j(\theta_j|\lambda_j)$
where $q_j(\theta_j|\lambda_j)=p\delta_0(\theta_j)+(1-p)\delta_{\lambda_j}(\theta_j)$
* i.e., each weight is either set to 0 with probability $p$ or set to with probability $1-p$
* This allows to interpret the result of ordinarily training a neural network with dropout as performing approximate Bayesian inference!

**Predictive Uncertainty via Dropout**
* Can approximate predictive uncertainty via
$p(y^*|\mathbf{x}^*,\mathbf{x}_{1:n},y_{1:n})\approx \mathbb{E}_{\theta\sim q(\cdot|\lambda)}[p(y^*|\mathbf{x}^*,\theta)]\approx \frac{1}{m} \sum_{j=1}^m p(y^*|\mathbf{x}^*,\theta^{(j)})$
* Hereby, each sample $\theta^{(j)}$ simply corresponds to a neural network with weights given by $\lambda$ , where each unit is set to 0 with probability $p$
* Thus, dropout is not only performed during training, but **also during prediction**!


**Probabilistic Ensembles of NNs**
* Another heuristic approach for approximate Bayesian inference with Neural Networks makes use of bootstrap sampling:
* Starting with dataset $D=\{(x_1,y_1),...,(x_n,y_n)\}$
* For $j=1:m$
    * Pick a data set $D_j$ of $n$ points uniformly at random from $D$ with replacement
    * Obtain MAP estimate (e.g., with SGD) on $D_j$ to obtain parameter estimate $\theta^{(j)}$
* Use approximate posterior:
$p(y^*|\mathbf{x}^*,\mathbf{x}_{1:n},y_{1:n})\approx \frac{1}{m} \sum_{j=1}^m p(y^*|\mathbf{x}^*,\theta^{(j)})$

**Overview**
* SVI / Bayes-by-Backprop
    * Optimizes ELBO via SGD (e.g., reparameterization gradients)
* Stochastic gradient MCMC techniques (SGLD, SGHMC)
    * Guaranteed to eventually converge to correct distribution
    * Need to summarize the MCMC iterates
* Monte-Carlo Dropout
    * Train model with dropout and SGD
    * Obtain predictive uncertainty via test-time dropout
* Probabilistic Ensembles
    * Train multiple models on random subsamples of the data
    * Sometimes a single model is trained with multiple ''heads'', each trained on different subsets of the data

**Aleatoric and Epistemic Uncertainty Beyond Regression**
* Standard approach for multi-class classification with NNs:
$\mathbf{p}=softmax(\mathbf{f})$
$p_i=\frac{exp(f_i)}{\sum_{j=1}^c exp(f_j)}$
$p(y|\mathbf{x};\theta)=p_y$
* Can explicitly model aleatoric uncertainty by **injecting learnable (Gaussian) noise** $\varepsilon$ and using $\mathbf{p}=softmax(\mathbf{f}+\varepsilon)$

**Evaluating Model Calibration**
* Can evaluate predictive distributions on held out data
Surpose training data $D_{train}\rightarrow$ variational posterior $q(\theta|\lambda)$
Consider validation data $D_{val}=\{(x_i',y_i')_{i=1:m}\}$
$\begin{aligned}
log P(y_{1:m}'|x_{1:m}',x_{1:n},y_{1:n})&\overset{i.i.d}=log \int P(y_{1:m}'|x_{1:m}',\theta)P(\theta|x_{1:n},y_{1:n})d\theta\\
&=log\int P(y_{1:m}'|x_{1:m}',\theta)q(\theta|\lambda)d\theta\\
&= log \mathbb{E}_{\theta\sim q_\lambda}P(y_{1:m}'|x_{1:m}',\theta)\\
&\overset{Jensen's}\geq \mathbb{E}_{\theta\sim q_\lambda}logP(y_{1:m}'|x_{1:m}',\theta)\\
&\approx \frac{1}{k}\sum_{j=1}^k \sum_{i=1}^m log P(y_i^*|x_i^*,\theta^{(j)})
\end{aligned}$
* *Note*: $\theta^{(j)}\sim q_\lambda, \quad$ $\sum_{i=1}^m log P(y_i^*|x_i^*,\theta^{(j)})$ is standard hold-out log-likelihood

**Estimating Calibration Error**
* **Partition test points into bins** according to predicted confidence values
* Then **compare accuracy** within a bin with average **confidence** within a bin
* Expected/maximum calibration error (ECE/MCE) is the average/maximum discrepancy across bins

**Improving Calibration**
* Can empirically **improve accuracy of calibration** via several heuristics
    * Histogram binning
    * Isotonic regression
    * Platt (temperature) scaling
    * ...

## Active Learning
### Lecture Notes
**Why is Uncertainty Useful?**
* So far, we have discussed several methods for probabilistic machine learning
* Key benefit: Modeling both **epistemic** and **aleatoric** uncertainty
* Will now discuss how to use of the uncertainty for deciding which data to collect
    * Active learning
    * Bayesian optimization

**Active Learning / Experiment Design**
* Suppose we’ve collected some data. Where should we sample to obtain most useful information?

**Recall: Bayesian Learning with GPs**
* Suppose $p(f)=GP(f;\mu;k)$
and we observe $y_i)=f(\mathbf{x}_i +\varepsilon_i)\quad \varepsilon_i\sim\mathcal{N}(0,\sigma^2)\quad A=\{\mathbf{x}_1,...,\mathbf{x}_m\}$
* Then $p(f|\mathbf{x}_1,...,\mathbf{x}_m,y_1,...,y_m)=GP(f;\mu',k)'$
where 
$\mu'(\mathbf{x})=\mu(\mathbf{x})+\mathbf{k}_{x,A}(\mathbf{K}_{AA}+\sigma^2\mathbf{I})^{-1}(\mathbf{y}_A-\mu_A)$
$k'(\mathbf{x},\mathbf{x}')=k(\mathbf{x},\mathbf{x}')-\mathbf{k}_{x,A}(\mathbf{K}_{AA}+\sigma^2\mathbf{I})^{-1}\mathbf{k}_{x',A}$
*Note*: $\mathbf{k}_{x,A}=[k(x,x_1),...k(x,x_m)]$
* **Posterior covariance $k’$ does not depend on $\mathbf{y}_A$**

**General Strategy**
* Query points whose observation **provides most useful information** about the unknown function
* How do we **quantify utility of an observation**?
* How do we find a **best set of observations to make**?

**Mutual Information / Information Gain**
* Given random variables $X$ and $Y$, the mutual information $I(X;Y)$ quantifies how much observing $Y$ reduces uncertainty about $X$, as measured by its entropy, in expectation over $Y$
* $I(X;Y)=H(X)-H(X|Y)$, where $H(X)$ is uncertainty before observing $Y$, and $H(X|Y)$ is uncertainty after observing $Y$.
Mutual information is symmetric: %I(X;Y)=I(Y;X)%
* $H(X)=\mathbb{E}_{X\sim p(x)}[-logp(x)]$
$H(X|Y)=\mathbb{E}_{Y\sim p(y)}[H(X|Y=y)]$
$H(X)+H(Y|X)=H(X,Y)$
E.g. $X\sim \mathcal{N}(\mu,\Sigma)\rightarrow H(X)=\frac{1}{2}ln(2\pi e)^d|\Sigma|$
* E.g. $X\sim\mathcal{N}(\mu,\Sigma), Y=X+\varepsilon,\varepsilon\sim\mathcal{N}(0,\sigma^2I)$
$\begin{aligned}
I(X;Y)&=H(Y)-H(Y|X)=H(Y)-H(\varepsilon)\\
&=\frac{1}{2}ln(2\pi e)^d|\Sigma+\sigma^2I|-ln(2\pi e)^d|\sigma^2I|\\
&=\frac{1}{2}ln|I+\sigma^{-2}\Sigma|
\end{aligned}$

**How do we quantify utility? Information Gain** [c.f., Lindley ‘56]
* Set $D$ of points to observe $f$ at
* Find $S \subseteq D$ maximizing information gain
$F(S):=H(f)-H(f|y_S)=I(f;y_S)=\frac{1}{2}log|I+\sigma^{-2}K_S|$
$H(f)$ is Uncertainty of $f$ before evaluation
$H(f|y_S)$ is Uncertainty of $f$ after evaluation
$y_S$ is Noisy obs. at locations $S$

**Optimizing Mutual Information** [cf Lindley ’56, Shewry & Wynn ’87]
* Mutual information $F(S)$ is NP-hard to optimize
* Simple strategy: **Greedy algorithm**. For $S_t=\{x_1,...,x_t\}$
$\begin{aligned}
x_{t+1}&=argmax_{x\in D}F(S_t\bigcup \{x\})-F(S_t)\\
&=argmax_{x\in D}H(y_{S_t+x})-H(y_{S_t+x}|f)-H(y_{S_t})+H(y_{S_t}|f)\\
&=argmax_{x\in D}H(y_x|y_{S_t})-H(y_x|f) \leftarrow \text{Constant for fixed noise variance}\\
&=argmax_{x\in D}\frac{1}{2}ln(2\pi e)\sigma^2_{x|S_t}-\frac{1}{2}ln(2\pi e)\sigma^2_n\\
&=argmax_{x\in D}\sigma^2_{x|S_t} \leftarrow \text{Entropy is monotonic in variance}
\end{aligned}$

**Active Learning: Uncertainty Sampling**
* Pick: $x_t=argmax_{x\in D}\sigma_{t-1}^2(x)$
where $\sigma_{t-1}^2(x):=\sigma_{x|x_{1:t-1}}^2$
* How good is the resulting design?

**Submodularity of Mutual Information**
* Mutual information $F(S)$ is **monotone submodular**: $B=A\bigcup A_c$
$\forall x\in D \: \forall A\subseteq B\subseteq D:F(A\bigcup\{x\})-F(A)\geq F(B\bigcup{x})-F(B)$
*Note*: $F(A\bigcup\{x\})-F(A)=H(y_x|y_A)-H(y_x|f)$, $H(\varepsilon)$ ind. of x for homoscedastic case.
$\Rightarrow F(A\bigcup\{x\})-F(A)\geq F(B\bigcup{x})-F(B)$
$\Leftrightarrow H(y_x|y_A)-H(y_x|f)\geq H(y_x|y_B)-H(y_x|f)$
$\Leftrightarrow H(y_x|y_A)\geq H(y_x|y_B)=H(y_x|y_A,y_{A_c})$
Surpose RVs $S,T,U$, it holds that $H(S|T)\geq H(S|T,U)$, called ''information never heard''
* I.e., satisfies **diminishing returns** property
* Greedy algorithm provides **constant-factor approximation** [Nemhauser et al’78]
$F(S_T)\geq (1-\frac{1}{e}max_{S\subseteq D,|S|\leq T}F(S)$
* I.e., uncertainty sampling is **near-optimal**!

**Failure of Uncertainty Sampling in Heteroscedastic Case**
* Uncertainty sampling **fails to distinguish epistemic and aleatoric uncertainty**
* In the heteroscedastic case, most **uncertain outcomes are not necessarily most informative** $P(y|f,x)=\mathcal{N}(y;f(x),\sigma^2(x))$
* Natural generalization: maximize
$x_{t+1}\in argmax_x\frac{\sigma_f^2(x)}{\sigma_n^2(x)}$
where $\sigma_f^2(x)$ is Epistemic uncertainty, and $\sigma_n^2(x)$ is Aleatoric uncertainty
*Note*: $I(X;Y)=\frac{1}{2}ln(2\pi e)\sigma_p^2-\frac{1}{2}ln(2\pi e)\sigma_n^2=\frac{1}{2}ln\frac{\sigma_f^2(x)}{\sigma_n^2(x)}$

**Outlook: Other Active Learning Objectives**
* Instead of entropy to quantify uncertainty, can derive alternative criteria à **Bayesian experimental design**
* For Gaussians, common choices scalarize the posterior covariance matrix in different ways
    * D-optimal design: entropy = log-determinant (= unc. samp.)
    * A-optimal design: trace
    * E-optimal design: maximal eigenvalue
    * ...
* These are typically more expensive to compute, but may offer other advantages
(e.g., A-optimal design minimizes the expected Mean-Squared Error)

**Active Learning for Classification**
* While we focused on regression, one can apply active learning also for other settings, such as classification § Here, uncertainty sampling corresponds to selecting samples that **maximize entropy of the predicted label**
$x_{t+1}\in argmax_x H(Y|x,x_{1:t},y_{1:t})$
where $H(Y|x,x_{1:t},y_{1:t})=-\sum_y logp(y|x,x_{1:t},y_{1:t})$
* While often a useful heuristic, also here, most uncertain label is not necessarily most informative

**Informative Sampling for Classification (BALD)**
Consider Bayesian learning with prior $p(\theta)$ over model params.(e.g. weights of some NN)
$p(y|x\theta)\propto exp(f_y(x,\theta))$
pick $x_{t+1}\in argmax_x I(\theta;y_x|x_{1:t},y_{1:t})=H(y|x,x_{1:t},y_{1:t})-\mathbb{E}_{\theta\sim p(\cdot|x_{1:t},y_{1:t})}H(y|x,\theta)$
where $H(y|x,x_{1:t},y_{1:t})$ is entropy of the perdictive distribution acc. to our (approx.) posterior,
$\mathbb{E}_{\theta\sim p(\cdot|x_{1:t},y_{1:t})}H(y|x,\theta)$ can approximate by sampling $\theta$ from posterior

**Summary Active Learning**
* Active learning refers to a family of approaches that aim to collect **data that maximally reduces uncertainty** about the unknown model
* For Gaussian processes and **homoscedastic** noise, uncertainty sampling is equivalent to greedily maximizing mutual information
* In general, need to account for epistemic and aleatoric uncertainty (optimize their ratio / BALD)
* Due to **submodularity**, **greedy algorithm** selects **near-optimal sets of observations**

## Bayesian Optimization
### Lecture Notes
**Exploration—Exploitation Tradeoffs**
* Numerous applications require trading experimentation (**exploration**) and optimization (**exploitation**)
* Often:
    * #alternatives >> #trials
    * experiments are noisy & expensive
    * similar alternatives have similar performance
* How can we exploit this regularity?

**Bayesian Optimization [Močkus et al. ’78, Jones ‘98, …]**
$x_t\to y_t=f(x_t)+\epsilon_t$
* How should we sequentially pick $x_1,...,x_T$ to find $max_x f(x)$ with minimal samples?

**Multi-armed Bandits**
* How should we allocate T tokens to k “arms” to maximize our return?
* Beautiful theory on how to explore & exploit [Robins ’52, Gittins’79, Auer+ ’02, …]
* Key principle: **Optimism in the face of uncertainty**
* Very successful in applications (e.g., drug trials, scheduling, ads…)

**Learning to Optimize**
* **Given**: Set of possible inputs $D$; noisy black-box access to unknown function $f\in\mathcal{F},f:D\to \mathbb{R}$
* **Task**: Adaptively choose inputs $x_1,...,x_T$ from $D$ After each selection, observe $y_t=f(x_t)+\varepsilon_t$
* **Cumulative regret**: $R_T=\sum_{t=1}^T r_t=\sum_{t=1}^T (max_{x\in D}f(x)-f(x_t))$
$r_t$ is information regret
Sublinear if $R_T/T\to 0$
implies $max_t f(x_t)\to f(x^*)$

**Gaussian Process Bandit Optimization**
* Goal: Pick inputs $x_1,x_2,...$ s.t. $\frac{1}{T}\sum_{t=1}^T[f(x^*)-f(x_t)]\to 0$ called ''average regret''
* How should we pick samples to minimize our regret?

**Optimistic Bayesian Optimization with GPs**
* Key idea: Focus exploration on plausible maximizers (upper confidence bound ≥ best lower bound)

**Upper Confidence Sampling (GP-UCB)** [use in Bandits: e.g., Lai & Robbuins ’85, Auer+ ’02, Dani+ ’08, ...]
* Pick input that maximizes upper confidence bound:
$x_t=argmax_{x\in D}\mu_{t-1}(x)+\beta_t\sigma_{t-1}(x)$
How should we choose $\beta_t$?
* Naturally trades off exploration and exploitation Only picks plausible maximizers

**Bayesian Regret of GP-UCB**
* Theorem: Assuming $f\sim GP$, if we choose $\beta_t$ ''correctly''
$\frac{1}{T}\sum_{t=1}^T[f(x^*)-f(x_t)]=\mathcal{O}^*(\sqrt{\frac{\gamma_T}{T}})$
where $\gamma_T=max_{|S|\leq T} I(f;y_S)$
* Key concept: “maximum information gain” $\gamma_T$ determines the regret

**Information Capacity of GPs**
* Regret depends on how quickly we can gain information
$\gamma_T=max_{|S|\leq T} I(f;y_S)$
* Submodularity of mutual info. yields data-dependent bounds

**Info. Gain Bounds for Common Kernels**
* Theorem: For the following kernels, we have:
    * Linear: $\gamma_T=\mathcal{O}(dlogT)$
    * Squared-exponential: $\gamma_T=\mathcal{O}(logT)^{d+1}$
    * Matérn with $\nu>2$, $\gamma_T=\mathcal{O}(T^{\frac{d(d+1)}{2\nu+d(d+1)}}logT)$
* Guarantees sublinear regret / convergence

**Outlook: Frequentist Regret for GP-UCB**
* Theorem: assume $f\in\mathcal{H}_k$
$\frac{1}{T}\sum_{t=1}^T[f(x^*)-f(x_t)]=\mathcal{O}^*(\sqrt{\frac{\beta_t \gamma_t}{T}})$
$O(\|f\|^2_k+\gamma_Tlog^3T)$
where $\|f\|^2_k$ is ''Complexity'' of $f$ (RKHS norm), $\gamma_T=max_{|A|\leq T}I(f;y_A)$

**Side note: Optimizing the Acquisition Function**
* GP-UCB requires solving the problem
$x_t=argmax_{x\in D}\mu_{t-1}(x)+\beta_t\sigma_{t-1}(x)$
* **This is generally non-convex**
* In low-D, can use Lipschitz-optimization (DIRECT, etc.)
* In high-D, can use gradient ascent (based on random initialization)

**Alternative Acquisition Functions**
* Besides UCB, there exist a number of alternative acquisition criteria
    * Expected Improvement à homework
    * Probability of improvement
    * Information Directed Sampling
    * **Thompson sampling**

**Thompson Sampling**
* At iteration $t$, Thompson sampling draws
$\tilde{f}\sim P(f|x_{1:t},y_{1:t})$
and selects $x_{t+1}\in argmax_{x\in D} \tilde{f}(x)$
* The randomness in the realization of $\tilde{f}$ is sufficient to trade exploration and exploitation
* It is possible to establish regret bounds for Thompson sampling similar to those for UCB

**Outlook: Hyperparameter Estimation**
* So far, have assumed that the kernel and its parameters are known. What if we need to learn them?
* In principle can **alternate learning hyperparameters** (e.g., via marginal likelihood maximization) and **observation selection**
* In practice, there is a **specific danger of overfitting**
    * Data sets in BO / active learning are **small**
    * Data points are selected **dependent** on prior observations
* Potential solutions
    * “Being Bayesian” about the hyperparameters (i.e., placing a hyperprior on them, and marginalizing them out)
    * Occasionally simply selecting some points at random

**Outlook: BO beyond GPs**
* Even though we focused on GPs, the ideas generalize to **other Bayesian learning problems** (e.g., involving Bayesian neural networks)
    * For UCB, can obtain (heuristic) confidence intervals using approximate inference (variational approximation, MCMC, ensembles etc.)
    * For Thompson sampling, need to sample from the posterior distribution over models, and then optimize the sample

## Markov Decision Processes
### Lecture Notes
**New Topic: Probabilistic Planning**
* So far: Probabilistic inference in dynamical models
    * E.g.: Tracking a robot based on noisy measurements
* Next: How should we **control** the robot to accomplish some goal / perform some task?

**Markov Decision Processes**
* An (finite) MDP is specified by 
    * A set of **states** $X=\{1,...,n\}$
    * A set of **actions** $A=\{1,...,m\}$
    * **Transition probabilities**
    $P(x'|x,a)=Prob(Next \:state=x'|Action\: a \:in\: state\: x)$
    * A **reward function** $r(x, a)$
    Reward can be random with mean $r(x, a)$;
    Reward may depend on $x$ only or $(x, a, x’)$ as well.
* For now assume $r$ and $P$ are known!
* Want to choose actions to maximize reward

**Applications of MDPs**
* Robot action planning
* Elevator scheduling
* Manufactoring processes
* Network switching and routing
* Foundation for Reinforcement Learning

**Planning in MDPs**
* Deterministic Policy: $\pi: X\to A$
* Randomized Policy: $\pi: X\to P(A)$
* Incduces a Markov chainL $X_0,X_1,...,X_t,...$ with transition probabilities
$P(X_{t+1}=x'|X_t=x)=P(x'|x,\pi(x))$
For randomized policies: $P(X_{t+1}=x'|X_t=x)=\sum_a \pi(a|x)P(x'|x,a)$
* Expected value $J(\pi)=\mathbb{E}[r(X_0,\pi(X_0))+\gamma r(X_1,\pi(X_1))+\gamma^2 r(X_2,\pi(X_2))+...]$
where $\gamma\in [0,1)$ is discounted factor

**Computing the Value of a Policy**
* For a fixed policy define **value function**
$V^\pi(x)=J(\pi|X_0=x)=\mathbb{E}[\sum_{t=0}^ \infin \gamma^tr(X_t,\pi(X_t))|X_0=x]$
Recursion:
$\begin{aligned}
V^\pi(x)&=J(\pi|X_0=x)=\mathbb{E}[r(X_0,\pi(X_0))+\sum_{t=1}^ \infin \gamma^tr(X_t,\pi(X_t))|X_0=x]\\
&\overset{lin. \:of\: exp.}=r(x,\pi(x))+\mathbb{E}[\sum_{t=1}^ \infin \gamma^tr(X_t,\pi(X_t))|X_0=x]\\
&\overset{index\:shift}=r(x,\pi(x))+\gamma\mathbb{E}_{X_{1:\infin}}[\sum_{t=0}^ \infin \gamma^tr(X_{t+1},\pi(X_{t+1}))|X_0=x]\\
&\overset{iter.\:expect.}=r(x,\pi(x))+\gamma\mathbb{E}_{X_1=x'}[\mathbb{E}_{X_{2:\infin}}[\sum_{t=0}^ \infin \gamma^tr(X_{t+1},\pi(X_{t+1}))|X_1=x']|X_0=x]\\
&\overset{def.\:outer\:expect.}=r(x,\pi(x))+\gamma\sum_{x'}P(x'|x,\pi(x))\mathbb{E}[\sum_{t=0}^ \infin \gamma^tr(X_{t+1},\pi(X_{t+1}))|X_1=x']\\
&\overset{stationary}=r(x,\pi(x))+\gamma\sum_{x'}P(x'|x,\pi(x))\mathbb{E}[\sum_{t=0}^ \infin \gamma^tr(X_t,\pi(X_t))|X_0=x']\\
\end{aligned}$

**Solving for the Value of a Policy**
* $V^\pi(x)=r(x,\pi(x))+\gamma\sum_{x'}P(x'|x,\pi(x))V^\pi(x')$
$V^\pi=r^\pi+\gamma T^\pi V^\pi$
$\Rightarrow r^\pi=(I-\gamma T^\pi)V^\pi$
$\Rightarrow V^\pi=(I-\gamma T^\pi)^{-1}r^\pi$
* Can compute $V^\pi$ exactly by solving linear system!

**Fixed Point Iteration**
* Can (approximately) solve the linear system via fixed point iteration:
* Initialize $V^\pi_0$ (e.g., as 0)
* For $t=1:T$ do $V_t^\pi=r^\pi+\gamma T^\pi V_{t-1}^\pi$
$B^\pi:\mathbb{R}^n\to \mathbb{R}^n$,$B^\pi V=r^\pi+\gamma T^\pi V\Rightarrow B^\pi V^\pi=V^\pi$
$B^\pi $ is a contraction: 
$\|B^\pi V-B^\pi V'\|_\infin=\|r^\pi +\gamma T^\pi V-r^\pi -\gamma  T^\pi V'\|_\infin=\gamma\|T^\pi (V-V')\|_\infin$
$=\gamma max_x\|\sum_{x'}P(x'|x,\pi(x))(V(x)-V'(x))\| \leq \gamma\|V-V'\|_\infin$
$\Rightarrow \|V_t^\pi -V^\pi \|_\infin \leq \gamma^t\|V_0^\pi -V^\pi \|_\infin \leq \varepsilon$
suffices that $tln\gamma+ln\|V_0^\pi -V^\pi \|_\infin\leq ln\varepsilon \Rightarrow t\geq \frac{ln\frac{\|V_0^\pi -V^\pi \|_\infin}{\varepsilon}}{-ln\gamma}$
* Computational advantages, e.g., for sparse transitions

**Value Functions and Policies**
* Value function $V^\pi $
$V^\pi(x)=r(x,\pi(x))+\gamma\sum_{x'}P(x'|x,\pi(x))V^\pi(x')$
Every value function induces a policy
* Greedy policy w.r.t. $V$
$\pi_V(x)=argmax_a r(x,a)+\gamma\sum_{x'}P(x'|x,a)V(x')$
Every policy induces a value function
* Theorem (Bellman): 
Policy optimal $\Leftrightarrow$ greedy w.r.t. its induced value function!
$V^*(x)=max_a[r(x,a)+\gamma\sum_{x'}P(x'|x,a)V^*(x')]$

**Policy Iteration**
* Start with an arbitrary (e.g., random) policy $\pi$
* Until converged do:
    * Compute value function $V^\pi(x)$
    * Compute greedy policy $\pi_G$ w.r.t. $V^\pi$
    * Set $\pi \leftarrow \pi_G$
* Guaranteed to
    * Monotonically improve
    * Converge to an optimal policy $\pi^*$ in $O*(n^2m/(1-\gamma))$ iterations! [Ye '10]

**Alternative Approach**
* Recall (Bellman): For the optimal policy $\pi^*$ it holds
$V^*(x)=max_a[r(x,a)+\gamma\sum_{x'}P(x'|x,a)V^*(x')]$
* Compute $V^*$ using **fixed point/dynamic programming**:
$V_t(x)=$ Max. expected reward when starting in state $x$ and world ends in $t$ time steps
$V_0(x)=max_a r(x,a)$ 
$V_1(x)=max_a r(x,a)+\gamma\sum_{x'}P(x'|x,a)V_0(x')$
$V_{t+1}(x)=max_a r(x,a)+\gamma\sum_{x'}P(x'|x,a)V_t(x')$

**Value Iteration**
* Initialize $V_0(x)=max_a r(x,a)$
* For $t=1$ to $\infin$
    * For each $x,a$ let
    $Q_t(x,a)=r(x,a)+\gamma\sum_{x'}P(x'|x,a)V_{t-1}(x')$
    * For each $x$ let $V_t(x)=max_a Q_t(x,a)$
    * Break if $\|V_t-V_{t-1}\|_\infin=max_x|V_t(x)-V_{t-1}(x)|\leq \varepsilon$
* Then choose greedy policy w.r.t $V_t$
* Guaranteed to converge to $\varepsilon$-optimal policy!

**Convergence of Value Iteration**
* Main ingredient of proof: Bellman update is a **contraction**
$B^:\mathbb{R}\to\mathbb{R}$, $(B^*V)(x)=max_a r(x,a)+\gamma\sum_{x'}P(x'|x,a)V_{t-1}(x')$
Bellman's theorem: $B^*V^*=V^*$
* Consider $V,V'\in\mathbb{R}^n$
$\|B^*V-B^*V'\|_\infin=max_x|(B^*V)(x)-(B^*V')(x)|=max_x|max_a Q(x,a)-max_{a'}Q'(x,a')|$
$\leq max_x max_a|Q(x,a)-Q'(x,a)|=\gamma max_{x,a}|\sum_{x'}P(x'|x,a)(V(x')-V'(x'))\leq\gamma\|V-V'\|_\infin$
$\Rightarrow \|V_t-V^*\|_\infin\leq\gamma^t\|V_0-V^*\|_\infin$
*Note*: $|max_a f(a)-max_{a'} f'(a')|\leq max_a |f(a)-f'(a)|$

**Tradeoffs: Value vs Policy Iteration**
* Policy iteration
    * Finds exact solution in polynomial # iterations!
    * Every iteration requires computing a value function
    * Complexity per iteration: Need to compute $V^{\pi_t}$ by solving linear system.
* Value iteration
    * Finds $\varepsilon$-optimal solution in polynomial # ($O(ln\frac{1}{\varepsilon}$)) iterations
    * Complexity per iteration: $O(nms)$ where $s$ is # of states can be reached from $(x,a)$
* In practice, which works better depends on application
* Can combine ideas of both algorithms

**Recap: Ways for solving MDPs**
* Policy iteration:
    * Start with random policy $\pi$
    * Compute exact value function $V^\pi$ (matrix inversion)
    * Select greedy policy w.r.t. $V^\pi$ and iterate
* Value iteration
    * Solve Bellman equation using dynamic programming
    $V_t(x)=max_a r(x,a)+\gamma\sum_{x'}P(x'|x,a)V_{t-1}(x')$
* Linear programming

**MDP = Controlled Markov Chain**
* State fully observed at every time step
* Action $A_t$ controls transition to $X_{t+1}$

**POMDP = Controlled HMM**
* Only obtain noisy observations $Y_t$ of the hidden state $X_t$
* Very powerful model! 
* Typically extremely intractable

**POMDP = Belief-state MDP**
* Key idea: POMDP as MDP with enlarged state space:
* New states: **beliefs** $P(X_t|y_{1:t})$ in the original POMDP
$\mathcal{B}=\Delta(\{1,...,n\})=\{b:\{1,...,n\}\to [0,1],\sum_x b(x)=1\}$
At time $t$: pick action $a_t \to$ new state $X_{t+1}\sim P(\cdot|x_t,a_t)\to$ obs. $y_{t+1}\sim P(\cdot|x_{t+1})$
At time 0: $b_1=P(X_1)\in \Delta^n:=\{b\in\mathbb{R}^n:b_i\geq 0,\sum_i b_i\}$
* Actions: Same as original MDP
* Transition model:
    * Stochastic observation:
    $P(Y_{t+1}=y|b_t,a_t)=\sum_{x,x'}b_t(x)P(x'|x,a_t)P(y|x')$
    * State update (Bayesian filtering!) Given $b_t,a_t,y_{t+1}$
    $b_{t+1}(x')=P(X_{t+1}=x|y_{1:t+1})\overset{Bayesian\:filtering}= \frac{1}{Z}b_t(x)P(X_{t+1}=x'|X_t=x,a_t)P(y_{t+1}|x')$
* Reward function: $r(b_t,a_t)=\sum_x b_t(x)r(x,a_t)$

**Solving POMDPs**
* For finite horizon $T$, set of reachable belief states is finite (but exponential in $T$)
* Can calculate optimal action using dynamic programming

**Approximate solutions to POMDPs**
* Key idea: most belief states never reached
    * Discretize the belief space by sampling
    * **Point based methods**:
        * Point based value iteration (PBVI)
        * Point based policy iteration (PBPI)
    * May want to apply dimensionality reduction
* Alternative approach: **Policy gradients**

**Policy Gradient Methods**
* Assume **parametric form** of policy
$\pi(b)=\pi(b;\theta)$
* For each parameter the policy $\theta$ induces a Markov chain
* Can compute expected reward $J(\theta)$ by sampling
* Find optimal parameters through search (gradient ascent)
$\theta^*=argmax_\theta J(\theta)$
* Will revisit when discussing RL