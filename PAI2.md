
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Bayesian Deep Learning](#bayesian-deep-learning)
  - [Lecture Notes](#lecture-notes)
    - [Variational Inference for Bayesian neural networks](#variational-inference-for-bayesian-neural-networks)
    - [Markov-Chain Monte Carlo for Bayesian Neural Networks](#markov-chain-monte-carlo-for-bayesian-neural-networks)
    - [Specialized Inference Techniques for Bayesian Neural Networks](#specialized-inference-techniques-for-bayesian-neural-networks)

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