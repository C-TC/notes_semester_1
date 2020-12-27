TODO
distribution parts in bishop maybe

# Probabilistic Artificial Intelligience 
task:
1. reintepret the lecture and add important comments of Krause. Formulas in latex and give a space for detailed proof.
2. one section for reading and maybe essential questions in homework
3. yandex or berkley materials to complement

## Introduction and probability
### Lecture Notes
**Topics Covered**
* Probabilistic foundations of AI
* Bayesian learning (GPs, Bayesian deep learning, variational inference, MCMC)
* Bandits & Bayesian optimization
* Planning under uncertainty (MDPs, POMDPs)
* (Deep) Reinforcement learning
* Applications (in class and in project)


**Review:Probability**
* **Probability space** $(\Omega,\mathcal{F},\mathcal{P})$
* set of **atomic events** $\Omega$
* set of all **non-atomic events** $\mathcal{F}$
* $\mathcal{F}$is a $\sigma \text{-algebra}$(closed under complements and countable unions)
    * $\Omega \in \mathcal{F} $
    * $A \in \mathcal{F} \rightarrow \Omega \backslash A \in \mathcal{F}$
    * $A_1,...,A_n,... \in \mathcal{F}\rightarrow \bigcup_iA_i \in \mathcal{F}$
* **Probability measure** $\mathcal{P}:\mathcal{F} \rightarrow [0,1]$
    * for $A \in \mathcal{F}$, $P(A)$ is the probability that event A happens

**Probability Axioms**
* Normalization: $P(\Omega)=1$
* Non-negativity: $P(A)\geq 0 \text{ for all}A \in \mathcal{F}$
* $\sigma \text{-additivity}$:$$\forall A_1,...,A_n,... \in \mathcal{F}\text{ disjoint:}P(\bigcup_{I=1}^\infin A_i)=\sum_{I=1}^\infin P(A_i)$$

**Interpretation of Probabilities**
* Frequentist interpretation
    * $P(A)$ is relative frequency of $A$ in repeated experiments
    * Can be difficult to assess with limited data
* Bayesian interpretation
    * $P(A)$ is ''degree of belief'' $A$ that will occur
    * Where does this belief come from?
    * Many different flavors (subjective, objective, pragmatic, …)

**Random Variables**
*  Let $D$ be some set (e.g., the integers)
*  A random variable $X$ is a mapping $X:\Omega \rightarrow D$
*  For some $x \in D$, we say
$$P(X=x)=P({\omega \in \Omega : X(\omega)=x})\qquad\text{“probability that variable X assumes state x”}$$

**Specifying Probability Distributions through RVs** 
* **Bernoulli** distribution: “(biased) coin flips”$D=\{H,T\}$
Specify $P(X=H)=p$. Then $P(X=T)=1-p$.
*Note*: can identify atomic ev.$\omega$ with $\{X=H\}$,$\{X=T\}$
* **Binomial** distribution counts no. heads $S$ in $n$ flips
* **Categorical** distribution: “(biased) m-sided dice” $D=\{1,...,m\}$
Specify $P(X=i)=p_i$, s.t. $p_i\geq 0,\sum p_i=1$
* **Multinomial** distribution counts the number of
outcomes for each side for $n$ throws

**Joint Distributions**
* random vector $\mathbf{X}=[X_1(\omega),...,X_n(\omega)]$
* can specify $P(X_1=x_1,...,X_n=x_n)$ directly (atomic events are assignments $x_1,...,x_n$)
* **Joint Distribution** describes relationship among all variables

**Conditional Probability**
* Formal definition:$$P(a|b)=\frac{P(a\wedge b)}{P(b)}\text{ if }P(b)\neq0$$
* **Product rule** $P(a\wedge b)=P(a|b)P(b)$
* for distributions: $P(A,B)=P(A|B)P(B)$
(set of equations, one for each instantiation of $A,B$)
$\forall a,b:P(A=b,B=b)=P(A=a|B=b)\cdot P(B=b)$
* **Chain(product) rule** for multiple RVs:$X_1,..,X_n$
$P(X_1,..,X_n)=P(X_{1:n})=P(X_1)\cdot P(X_2|X_1)\cdot ... \cdot P(X_n|X_{1:n-1})$

**The Two Rules for Joint Distributions**
* **Sum rule (Marginalization)**
$P(X_{1:i-1},X_{i+1:n})=\sum_{x_i}P(X_{1:i-1},X_i=x_i,X_{i+1:n})$
* **Product rule (chain rule)**

**Bayes' Rule**
Given:
* **Prior** $P(X)$
* **Likelihood** $P(X|Y)=\frac{P(X,Y)}{P(Y)}$

Then:
* **Posterior**
$P(X|Y)=\frac{P(X)P(Y|X)}{\sum_{X=x}P(X=x)P(Y|X=x)}$

**Independent RVs**
* Random variables$X_1,...,X_n$ are called  **independent** if 
$P(X_1=x_1,...,X_n=x_n)=P(x_1)P(x_2)\dots P(x_n)$

**Conditional Independence**
* Rand. vars. $X$ and $Y$ conditionally independent given $Z$ **iff** for all $x,y,z$:
$P(X=x,Y=y|Z=z)=P(X=x|Z=z)P(Y=y|Z=z)$
* If $P(Y=y|Z=z)>0$, that is equivalent to 
$P(X=x|Y=y,Z=z)=P(X=x|Z=z)$
Similar for sets of random variables$\mathbf{X},\mathbf{Y},\mathbf{Z}$
we write:$\mathbf{X}\bot \mathbf{Y}|\mathbf{Z}$

**Problems with High-dim. Distributions**
* Suppose we have $n$ binary variables, then we have $2^{n-1}$ variables to specify
$P(X_1=x_1,..,X_n=x_n)$
* Computing marginals:
    * Suppose we have joint distribution $P(X_1,..,X_n)$
    * Then (acc. to sum rule)
    $$P(X_i=x_i)=\sum_{x_{1:i-1},x_{i+1:n}}P(x_1,...,x_n)$$
    * If all $X_i$ are binary: this sum has $2^{n-1}$ terms
* Conditional queries
    * Suppose we have joint distribution $P(X_1,..,X_n)$
    * Compute distribution of some variables given values for others:
    $P(X_1=\cdot |X_7=x_7)=\frac{P(X_1=\cdot ,X_7=x_7)}{P(X_7=x_7)}=\frac{1}{Z}P(X_1=\cdot ,X_7=x_7)$
    where, $Z=\sum_{x_1}P(X_1=x_1,X_7=x_7)$
    where, $P(X_1=x_1,X_7=x_7)=\sum_{x_{2:6}}\sum_{x_{8:n}}P(X_{1:n}=x_{1:n})$ , $2^{n-2}$ terms for binomial $X_i$
* Representation (parametrization)
* Learning (estimation)
* Inference (prediction)

**Gaussian Distribution**
* univariate : 
$$p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
$\sigma$: Std. dev., $\mu$: mean
* multivaraite:
$$p(\mathbf{x})=\frac{1}{2\pi \sqrt{|\Sigma|}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$$
where $\Sigma=\begin{pmatrix}\sigma_1^2&\sigma_{12}\\\sigma_{21}&\sigma_2^2\end{pmatrix}$ , $\mu=\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix}$.
* **Multivariate Gaussian distribution**
$$\mathcal{N}(y;\Sigma,\mu)=\frac{1}{((2\pi)^{n/2} \sqrt{|\Sigma|}}exp(-\frac{1}{2}(y-\mu)^T\Sigma^{-1}(y-\mu))$$
where $\Sigma=\begin{pmatrix}\sigma_1^2&\sigma_{12}&...&\sigma_{1n}\\\vdots&\:&\:&\vdots\\\sigma_{n1}&\sigma_{n2}&...&\sigma_n^2\end{pmatrix}$ ,
$\sigma_{ij}=\mathbb{E}((x_i-\mu_i)(x_j-\mu_j))$ , 
$\sigma_i^2=\mathbb{E}((x_i-\mu_i)^2)=Var(x_i)$.
The joint distribution over $n$ variables requires **only $O(n^2)$ parameters**.
* **Fact:Gaussians are independent iff they are uncorrelated:**
$X_i\bot X_j \Leftrightarrow \sigma_{ij}=0$
* Multivariate Gaussians have important properties:
    * **Compact representation** of high-dimensional joint distributions
    * **Closed form inference**

**Bayesian Inference in Gaussian Distributions**
* Suppose we have a Gaussian random vector
$\mathbf{X}=\mathbf{X}_V=[X_1,...,X_d]\sim \mathcal{N}(\mu_V,\Sigma_{VV})$
* Hereby $V=\{1,...,d\}$ is an index set.
* Suppose we consider a subset of the variables
$A=\{i_1,...,i_k\}, \quad i_j \in V$
* The **marginal distribution** of variables indexed by $A$ is:
$\mathbf{X}_A=[X_{i_1},...,X_{i_k}]\sim \mathcal{N}(\mu_A,\Sigma_{AA})$
where $\mu_A=[\mu_{i_1},...,\mu_{i_k}]$ , $\Sigma_{AA}=
\begin{pmatrix}
\sigma_{i_1 i_1}&...&\sigma_{i_1 i_k}\\
\vdots&\ddots&\vdots
\\\sigma_{i_k i_1}&...&\sigma_{i_k i_k}
\end{pmatrix}$

**Conditional Distributions**
* Suppose we have a Gaussian random vector
$\mathbf{X}=\mathbf{X}_V=[X_1,...,X_d]\sim \mathcal{N}(\mu_V,\Sigma_{VV})$
* Further, suppose we take two disjoint subsets of $V$
$A=\{i_1,...,i_k\}\quad B=\{j_1,...,j_m\}$
* The **conditional distribution**
$p(\mathbf{X}_A|\mathbf{X}_B=\mathbf{x}_B)=\mathcal{N}(\mu_{A|B},\Sigma_{A|B})$
is Gaussian, **where**
$$\mu_{A|B}=\mu_A+\Sigma_{AB}\Sigma_{BB}^{-1}(\mathbf{x}_B-\mu_B)$$
$$\Sigma_{A|B}=\Sigma_{AA}-\Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}$$
where $\Sigma_{AB}=
\begin{pmatrix}
\sigma_{i_1 j_1}&...&\sigma_{i_1 j_m}\\
\vdots&\ddots&\vdots
\\\sigma_{i_k j_1}&...&\sigma_{i_k j_m}
\end{pmatrix}\in \mathbb{R}^{k\times m}$

**Multiples of Gaussians are Gaussian**
* Suppose we have a Gaussian random vector
$\mathbf{X}=\mathbf{X}_V=[X_1,...,X_d]\sim \mathcal{N}(\mu_V,\Sigma_{VV})$
* Take a matrix $M \in \mathbb{R}^{m\times d}$
* Then the random vector $\mathbf{Y}=\mathbf{MX}$ is Gaussian:
$$\mathbf{Y} \sim \mathcal{N}(\mathbf{M}_{\mu_V},\mathbf{M}\Sigma_{VV}\mathbf{M}^T$$

**Sums of Gaussians are Gaussian**
* Suppose we have independent two Gaussian random vectors
$\mathbf{X}=\mathbf{X}_V=[X_1,...,X_d]\sim \mathcal{N}(\mu_V,\Sigma_{VV})$
$\mathbf{X}'=\mathbf{X}'_V=[X'_1,...,X'_d]\sim \mathcal{N}(\mu'_V,\Sigma'_{VV})$

## Bayesian Learning
### Lecture Notes
**Recall: linear regression**
* $y\approx \mathbf{w}^T\mathbf{x}=f(\mathbf{x})$

**Recall: ridge regression**
* Regularized optimization problem:
$\min_\mathbf{w}\sum_i(y_i-\mathbf{w}^T\mathbf{x}_i)^2+\lambda\|\mathbf{w}\|_2^2$
* Can optimize using (stochastic) gradient descent, or still find **analytical solution**:
$\hat{\mathbf{w}}=(\mathbf{X}^T\mathbf{X}+\lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

**Ridge regression as Bayesian inference**
Assume $p(\mathbf{w})=\mathcal{N}(0,\sigma_p^2\cdot \mathbf{I}))$ independent of $\mathbf{x}_{1:n}$
conditional iid. $\Rightarrow$ $p(y_{1:n}|\mathbf{w},\mathbf{x}_{1:n})=\prod_{i=1}^n p(y_i|\mathbf{w},\mathbf{x}_i)$
In particular: $p(y_i|\mathbf{w},\mathbf{x}_i)=\mathcal{N}(y_i;\mathbf{w}^T\mathbf{x}_i,\sigma_n^2)\Leftrightarrow y_i=\mathbf{w}^T\mathbf{x}_i+\varepsilon_i\quad \varepsilon_i\sim \mathcal{N}(0,\sigma_n^2)$
Then, $\begin{aligned}
p(\mathbf{w}|\mathbf{x}_{1:n},\mathbf{y}_{1:n})&=\frac{1}{Z}p(\mathbf{w}|\mathbf{x}_{1:n})p(\mathbf{y}_{1:n}|\mathbf{w},\mathbf{x}_{1:n})\\
&=\frac{1}{Z}p(\mathbf{w})\prod_{i=1}^n p(y_i|\mathbf{w},\mathbf{x}_i)\\
&=\frac{1}{ZZ_p}exp(-\frac{1}{2\sigma_p^2}\|\mathbf{w}\|^2_2)\frac{1}{Z_l}\prod exp(-\frac{1}{\sigma_n^2}(\mathbf{y}_i-\mathbf{w}^T\mathbf{x}_i)^2)\\
&=\frac{1}{Z'}exp(-\frac{1}{2\sigma_p^2}\|\mathbf{w}\|^2_2-\frac{1}{\sigma_n^2}(\mathbf{y}_i-\mathbf{w}^T\mathbf{x}_i)^2)
\end{aligned}$
$\Rightarrow \argmax_\mathbf{w}p(\mathbf{w}|\mathbf{x}_{1:n},y_{1:n})=\argmin_\mathbf{w}\sum_{i=1}^n(y_i-\mathbf{w}^T\mathbf{x}_i)^2+\lambda\|\mathbf{w}\|_2^2$
$\qquad\rightarrow \lambda=\frac{\sigma_n^2}{\sigma_p^2}$

**Ridge regression = MAP estimation**
* Ridge regression can be understood as finding the **Maximum A Posteriori (MAP) parameter estimate** for a linear regression problem, assuming that
* The **noise** $P(y|\mathbf{x},\mathbf{w})$ is **(cond.) iid Gaussian** and
* The **prior** $P(\mathbf{w})$ on the model parameters $\mathbf{w}$ is **Gaussian**
* However, ridge regression returns a single model
* Such a **point estimate** does not quantify **uncertainty**

**Bayesian Linear Regression (BLR)**
* Key idea: Reason about full posterior of $\mathbf{w}$, not only its mode
* For Bayesian linear regression with Gaussian prior and Gaussian likelihood, posterior has **closed form**

**Posterior distributions in BLR**
* Prior: $p(\mathbf{w}=\mathcal{N}(0,\mathbf{I})$
* Likelihood: $p(y|\mathbf{x},\mathbf{w},\sigma_n)=\mathcal{N}(y;\mathbf{w}^T\mathbf{x},\sigma_n^2)$
* Posterior: 
$p(\mathbf{w}|\mathbf{X},\mathbf{y})=\mathcal{N}(\mathbf{w};\bar{\mu},\bar{\Sigma})$
$\bar{\mu}=(\mathbf{X}^T\mathbf{X}+\sigma_n^2\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
$\bar{\Sigma}=(\sigma_n^{-2}\mathbf{X}^T\mathbf{X}+\mathbf{I})^{-1}$
* $\bar{\mu}$ is ridge regression solution!
* Precision matrix: $\bar{\Lambda}=\bar{\Sigma}^{-1}=\sigma_n^{-2}\mathbf{X}^T\mathbf{X}+\mathbf{I}$

**Making predictions in BLR**
* Define $f^*=\mathbf{w}^T\mathbf{x}^*$
$\to p(f^*|\mathbf{x}_{1:n},\mathbf{y}_{1:n},\mathbf{x}^*)=\int p(f^*|\mathbf{w},\mathbf{y}_{1:n},\mathbf{x}^*)p(\mathbf{w}|\mathbf{x}_{1:n},\mathbf{y}_{1:n},\mathbf{x}^*)d\mathbf{w}$
since $\mathbf{w}\sim\mathcal{N}(\bar{\mu},\bar{\Sigma}),\mathbf{y}^*=f^*+\varepsilon, \varepsilon\sim \mathcal{N}(0,\sigma_n^2)$
$p(f^*|\mathbf{X},\mathbf{y},\mathbf{x}^*)=\mathcal{N}(\bar{\mu}^T\mathbf{x}^*,{\mathbf{x}^*}^T\bar{\Sigma}\mathbf{x}^*)$
$p(y^*|\mathbf{X},\mathbf{y},\mathbf{x}^*)=\mathcal{N}(\bar{\mu}^T\mathbf{x}^*,{\mathbf{x}^*}^T\bar{\Sigma}\mathbf{x}^*+\sigma_n^2)$

**Aleatoric vs. epistemic uncertainty**
* Uncertainty about $f^*$ : $\bar{\Sigma}\leftarrow \text{(epistemic)}$
* Noise/Uncertainty about $y^*$ given $f^*$ : $\sigma_n^2\leftarrow \text{(aleatoric)}$
* Can distinguish two forms of uncertainty:
    * **Epistemic uncertainty**: Uncertainty about the model due to
the lack of data
    * **Aleatoric uncertainty**: Irreducible noise


## Bayesian Linear Regression(cont'd)
### Lecture Notes
* Observations: Conditional Linear Gaussians
* If $X$, $Y$ are jointly Gaussian, then $p(X|Y=y)$ is Gaussian, with mean linearly dependent on $y$:
$p(X=x|Y=y)=\mathcal{N}(x;\mu_{X|Y}\sigma_{X|Y}^2)$
$\mu_{X|Y}=\mu_X+\sigma_{XY}\sigma_Y^2(y-\mu_Y)$
* Thus random variable $X$ can be viewed as a linear function of $Y$ with independent Gaussian noise added
$X=a\cdot Y+b+\varepsilon$, where $a=\sigma_{XY}\sigma_{Y}^2$, $b=\mu_X-\sigma_{XY}\sigma_Y^2\mu_Y$
* The converse also holds.

**Ridge regression vs Bayesian lin. regression**
* Ridge regression: predict using *MAP estimate* for weights
$\hat{\mathbf{w}}=\argmax_\mathbf{w}p(\mathbf{w}|\mathbf{x}_{1:n},y_{1:n})$
$p(y^*|\mathbf{x}^*,\hat{\mathbf{w}})=\mathcal{N}(y^*;\hat{\mathbf{w}}^T\mathbf{x}^*,\sigma_n^2)$
* BLR: predict by averaging all $\mathbf{w}$ acc. to posterior:
$p(y^*|\mathbf{X},\mathbf{y},\mathbf{x}^*)
=\int p(y^*|\mathbf{x}^*,\mathbf{w})p(\mathbf{w}|\mathbf{x}_{1:n},\mathbf{y}_{1:n})d\mathbf{w}
=\mathcal{N}(\bar{\mu}^T\mathbf{x}^*,{\mathbf{x}^*}^T\bar{\Sigma}\mathbf{x}^*+\sigma_n^2)$
* Thus, ridge regression can be viewed as approximating the full posterior by **(placing all mass on) its mode**
$p(y^*|\mathbf{X},\mathbf{y},\mathbf{x}^*)
=\int p(y^*|\mathbf{x}^*,\mathbf{w})p(\mathbf{w}|\mathbf{x}_{1:n},\mathbf{y}_{1:n})d\mathbf{w}$
$\approx \int p(y^*|\mathbf{x}^*,\mathbf{w})\delta_{\hat{\mathbf{w}}}(\mathbf{w})d\mathbf{w}$
$=p(y^*|\mathbf{x}^*,\hat{\mathbf{w}})$
$\hat{\mathbf{w}}=\argmax_\mathbf{w}p(\mathbf{w}|\mathbf{x}_{1:n},y_{1:n})$
* *Note*: $\delta_{\hat{\mathbf{w}}}(\cdot)$ is such that $\int f(\mathbf{w})\delta_{\hat{\mathbf{w}}}(\mathbf{w})d\mathbf{w}=f(\hat{\mathbf{w}})$

**Choosing hyperparameters** 
* In BLR, need to specify the (co-)variance of the prior $\sigma_p$ and the variance of the noise $\sigma_n$
* These are **hyperparameters** of the model (governing the distribution of the parameters $\mathbf{w}$)
* How to choose? One option:
    * Choose $\hat{\lambda}=\frac{\hat{\sigma}_n^2}{\hat{\sigma}_p^2}$ via cross-validation
    * Then estimate $\hat{\sigma}_n^2=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{\mathbf{w}}^T\mathbf{x}_i)^2$ as the empirical variance of the residual, and solve for $\hat{\sigma}_p^2=\frac{\hat{\sigma}_n^2}{\hat{\lambda}}$
* Another option: marginal likelihood of the data, see **Gaussian Process (marginal likelihood)**

**Side note: Graphical models**
* Have seen: Can represent arbitrary joint distributions as product of conditionals via chain rule
* Often, factors only depend on subsets of variables
* Can represent the resulting product as a directed acyclic graph
* Graphical model for BLR (see lecture notes)

**Recursive Bayesian updates**
* “Today’s posterior is tomorrow’s prior”
* Surpose that:
Prior: $p(\theta)$, observe $y_{1:n}$, s.t. $p(y_{1:n}|\theta)=\prod_{i=1}^n p_i(y_i|\theta)$
for BLR: $\theta \equiv \mathbf{w}$, $ p_i(y_i|\theta) \equiv p(y_i|\mathbf{w},\mathbf{x}_i)$
Define $p^{(j)}(\theta)$ to be the posterior afer recurring the first $j$ observation. $p^{(j)}(\theta)=p(\theta|y_{1:j})$
* $p^{(0)}(\theta)=p(\theta)=\mathcal{N}(0,\sigma_p \cdot \mathbf{I})$
Surpose we have cumputed $p^{(j)}(\theta) \equiv \mathcal{N}(\mu^{(j)},\Sigma^{(j)})\leftarrow$ posterior $\theta^{(j)}=\{\mu^{(j)},\Sigma^{(j)}\}$ 
and observed $y_j$.
* $p^{(j+1)}(\theta)=p(\theta|y_{1:j+1})=\frac{1}{Z} p(\theta|y_{1:j})p(y_{j+1}|\theta,y_{1:j})=\mathcal{N}(\mu^{(j+1)},\Sigma^{(j+1)})$
where, $\theta^{(j+1)}=\{\mu^{(j+1)},\Sigma^{(j+1)}\},\quad p(\theta|y_{1:j})=p^{(j)}(\theta),\quad p(y_{j+1}|\theta,y_{1:j})=p_{j+1}(y_{j+1}|\theta)$

**Summary Bayesian Linear Regression**
* **Bayesian linear regression** makes same modeling assumptions as ridge regression (Gaussian prior on weights, Gaussian noise)
* BLR computes / uses **full posterior distribution** over the weights rather than the mode only
* Thus, it captures **uncertainty in weights**, and allows to separate epistemic from aleatoric uncertainty
* Due to independence of the noise, can do **recursive updates** on the weights


## Kalman Filters
### Lecture Notes
**Kalman filters**
* Track objects over time using noisy observations
    * E.g., robots moving, industrial processes,...
* State described using **Gaussian variables**
    * E.g., location, velocity, acceleration in 3D
* Assume conditional linear Gaussian dependencies for states and observations

**Kalman Filters: The Model**
* $X_1,...,X_T$: Location of object being tracked
* $Y_1,...,Y_T$: Observations
* $P(X_1)$: **Prior** belief about location at time 1 (Gaussian)
* $P(X_{t+1}|X_t)$: **Motion Model**
    * How do I expect my target to move in the environment?
    $\mathbf{X}_{t+1}=\mathbf{F}\mathbf{X}_t+\varepsilon_t$, where $\varepsilon_t \in \mathcal{N}(0,\Sigma_x)$
* $P(Y_t|X_t)$: **Sensor model** 
    * What do I observe if target is at location $X_t$?
    $\mathbf{Y}_t=\mathbf{H}\mathbf{X}_t+\eta_t$, where $\eta_t \in \mathcal{N}(0,\Sigma_y)$
* Assumptions:
Known: $\mathbf{X}_{t+1}=\mathbf{F}\mathbf{X}_t+\varepsilon_t$, $\mathbf{Y}_t=\mathbf{H}\mathbf{X}_t+\eta_t$,$\qquad\varepsilon_{1:t},\eta_{1:t}$ independent
implies that: $X_{t+1}\bot X_{1:t-1}|X_t$, and $Y_{t+1}\bot Y_{1:t-1},X_{1:t-1}|X_t$
$\rightarrow P(X_{1:t},Y_{1:t})=P(X_1)P(X_2|X_1)\dots P(X_n|X_{n-1})P(Y_1|X_1)P(Y_2|X_2)\dots P(Y_n|X_n)$
$=P(X_1)P(Y_1|X_1)\prod_{i=2}^tP(X_i|X_{i-1})P(Y_i|X_i)$

**Bayesian filtering**
* Start with $P(X_1)=\mathcal{N}(\mu,\Sigma)$
* At time $t$
    * Assume we have $P(X_t|Y_{1,...,t-1})$
    * **Conditioning**: $P(X_t|Y_{1,...t})=\frac{1}{Z}P(X_t|Y_{1:t-1})P(Y_t|X_t,Y_{1:t-1})$, where $P(Y_t|X_t,Y_{1:t-1})=P(Y_t|X_t)$, so that $Z=\int P(X_t|Y_{1:t-1})P(Y_t|X_t)dX_t$
    * **Prediction**: $P(X_{t+1}|Y_{1,...t})=\int P(X_{t+1},X_t|Y_{1:t})dX_t=\int P(X_{t+1}|X_t,Y_{1:t})P(X_t|Y_{1:t})dX_t=\int P(X_{t+1}|X_t)P(X_t|Y_{1:t})dX_t$
    * For Gaussians, can compute these integrals in closed form!
* Example: Random walk in 1D
    * Transition / motion model: $P(x_{t+1}|x_t)=\mathcal{N}(x_t,\sigma_x^2)$
    $x_{t+1}=x_t+\varepsilon_t$, $\quad \varepsilon_t\sim \mathcal{N}(0,\sigma_x^2)$
    * Sensor model: $P(y_t|x_t)=\mathcal{N}(x_t,\sigma_y^2)$
    $y_t=x_t+\eta_t$, $\quad \eta_t\sim \mathcal{N}(0,\sigma_y^2)$
    * State at time t: $P(x_t|y_{1:t})=\mathcal{N}(\mu_t,\sigma_t^2)$
    * $\rightarrow \mu_{t+1}=\frac{\sigma_y^2\mu_t+(\sigma_t^2+\sigma_x^2)y_{t+1}}{\sigma_t^2+\sigma_x^2+\sigma_y^2}\quad \sigma_{t+1}=\frac{(\sigma_t^2+\sigma_x^2)\sigma_y^2}{\sigma_t^2+\sigma_x^2+\sigma_y^2}$

**General Kalman update**
* Transition model: $P(\mathbf{x}_{t+1}|\mathbf{x}_t)=\mathcal{N}(\mathbf{x}_{t+1};\mathbf{F}\mathbf{x}_t,\Sigma_x)$
* Sensor model: $P(\mathbf{y}_t|\mathbf{x}_t)=\mathcal{N}(\mathbf{y}_t;\mathbf{H}\mathbf{x}_t,\Sigma_y)$
* **Kalman Update**: 
$\mu_{t+1}=\mathbf{F}\mu_t+\mathbf{K}_{t+1}(\mathbf{y}_{t+1}-\mathbf{H}\mathbf{F}\mu_t)$
$\mathbf{\Sigma}_{t+1}=(\mathbf{I}-\mathbf{K}_{t+1}\mathbf{H})(\mathbf{F}\mathbf{\Sigma}_t\mathbf{F}^T+\mathbf{\Sigma_x})$
* **Kalman Gain**:$$\mathbf{K}_{t+1}=(\mathbf{F}\mathbf{\Sigma}_t\mathbf{F}^T+\mathbf{\Sigma_x})\mathbf{H}^T(\mathbf{H}(\mathbf{F}\mathbf{\Sigma}_t\mathbf{F}^T+\mathbf{\Sigma_x})\mathbf{H}^T+\mathbf{\Sigma}_y)^{-1}$$
* Can compute $\mathbf{\Sigma}_t$ and $\mathbf{H}_t$ **offline**

**BLR vs Kalman Filtering**
* Can view Bayesian linear regression as a form of a Kalman filter!
    * Hidden variables are the weights
    * Forward model is constant (identity)
    * Observation model at time $t$ is determined by data point $x_t$

## Gaussian Process
### Lecture Notes
**What about nonlinear functions?**
* Recall: Can apply linear method (like BLR) on nonlinearly transformed data. However, computational cost increases with dimensionality of the feature space!
$f(\mathbf{x})=\sum_{i=1}^dw_i\phi_i(\mathbf{x})$
In $d$-dim,: $\mathbf{x}=[x_1,...,x_d]$, $\Phi(\mathbf{x})=[1,x_1,...,x_d,x_1^2,...,x_d^2,x_1x_2,...,x_{d-1}x_d,...,x_1\cdot...\cdot x_m,...,x_{d-m+1}\cdot ... \cdot x_d]\leftarrow O(d^m)$ monomials of deg $m$

**The ''Kernel Trick''**
* Express problem s.t. it only depends on inner products
* Replace inner products by kernels
* $\mathbf{x}_i^T\mathbf{x}_j \Rightarrow k(\mathbf{x}_i,\mathbf{x}_j)$
* $\Phi(\mathbf{x})=[\text{all monomials of deg }\leq m]$
$\Rightarrow k(\mathbf{x},\mathbf{x}')=(\mathbf{x}^T\mathbf{x}'+1)^m\quad \text{implicitly represents all monimials o f degree up to }m$

**Weight vs Function Space View**
* Assume **Gaussian prior** on the weights: $\mathbf{w}\in \mathbb{R}^d\sim\mathcal{N}(0,\sigma_p^2\mathbf{I})$
* This imply **Gaussian distribution on the predictions**
* Suppose we consider an arbitrary (finite) set of inputs $\mathbf{X}=\begin{pmatrix}\mathbf{x}_1\\\vdots\\\mathbf{x}_n
\end{pmatrix} \in \mathbb{R}^{n\times d}$
* The predictive distribution is given by:
    * $f\sim \mathcal{N}(0,\sigma_p^2\mathbf{X}\mathbf{X}^T)\leftarrow \text{let }\mathbf{K}_{ij}=\mathbf{x}_i^T\mathbf{x}_j,\mathbf{K}\in \mathbb(R)^{n\times n}$
    * where $f=[f_1,...,f_n],f_i=\mathbf{x}_i^T\mathbf{w}\rightarrow f=\mathbf{Xw}$

**Predictions in “function space”**
* Suppose we’re given data $\mathbf{X}$, $\mathbf{y}$, and want to predict $\mathbf{x}^*$
    * $\widetilde{\mathbf{X}}=\begin{pmatrix}\mathbf{X}\\\mathbf{x}^*
    \end{pmatrix}$,$\quad \tilde{\mathbf{y}}=
    \begin{pmatrix}\mathbf{y}\\y^*
    \end{pmatrix}$,$\quad \tilde{\mathbf{f}}=
    \begin{pmatrix}\mathbf{f}\\f^*
    \end{pmatrix} $
    $\rightarrow \tilde{\mathbf{f}}=\widetilde{\mathbf{X}}\cdot \mathbf{w},\tilde{\mathbf{y}}=\tilde{\mathbf{f}}+\tilde{\mathbf{\varepsilon}},\tilde{\mathbf{\varepsilon}}\sim \mathcal{N}(0,\sigma_n^2\mathbf{I}_{n+1})$
* $\rightarrow \tilde{\mathbf{y}}\sim \mathcal{N}(0,\widetilde{\mathbf{X}}\widetilde{\mathbf{X}}^T+\sigma_n^2\mathbf{I})$ ,where $\widetilde{\mathbf{K}}=\widetilde{\mathbf{X}}\widetilde{\mathbf{X}}^T$
* $\rightarrow P(y^*|\mathbf{x}_{1:n},\mathbf{y}_{1:n})=\mathcal{N}(\mu_{\mathbf{x}^*|\mathbf{x}_{1:n},\mathbf{y}_{1:n}},\sigma_{\mathbf{x}^*|\mathbf{x}_{1:n}}^2)$

**Key Insight**
* For prior $\mathbf{w}\sim\mathcal{N}(0,\mathbf{I})$, the predictive distribution over $\mathbf{f}=\mathbf{Xw}$ is Gaussian
$\mathbf{f}\sim \mathcal{N}(0,\mathbf{X}\mathbf{X}^T)\equiv \mathcal{N}(0,\mathbf{K})$
* Thus, data points only enter as inner products!
* Can kernelize: $\mathbf{f}\sim\mathcal{N}(0,\mathbf{K})$ , where $\mathbf{K}_{\mathbf{x},\mathbf{x}'}=\phi(\mathbf{x})^T\phi(\mathbf{x}')=\mathbf{k}(\mathbf{x},\mathbf{x}')$
e.g. poly. kernel $(1+\mathbf{x}^T\mathbf{x}')^m$

**What about infinite domains?**
* The previous construction can be generalized to **infinitely large domains $\mathbf{X}$**
* The resulting random function is called a **Gaussian process**

**Bayesian learning with Gaussian processes**
* c.f. Rasmussen & Williams 2006
* $Likelihood: P(data|f)\qquad Posterior:P(f|data)$
* Predictive uncertainty + tractable inference

**Gaussian Processes**
* $\infin$-dimension Gaussian
* Gaussian process (GP) = normal distribution over functions
* Finite marginals are multivariate Gaussians
* Closed form formulae for Bayesian posterior update exist
* Parameterized by covariance function $k(\mathbf{x},\mathbf{x}')=Cov(f(\mathbf{x}),f(\mathbf{x}'))$
* A **Gaussian Process (GP)** is an:
    * (infinite) set of random variables, indexed by some set $\mathbf{X}$
    i.e., there exists functions $\mu:X\rightarrow \mathbb{R}\quad k:X\times X \rightarrow \mathbb{R}$
    such that for all $A \subseteq X,\quad A=\{x_1,...,x_m\}$
    it holds that $Y_A=[Y_{x_1},...,Y_{x_m}]\sim \mathcal{N}(\mu_A,\mathbf{K}_{AA})$
    where,
    $\mathbf{K}_{AA}=\begin{pmatrix}k(x_1,x_1)&k(x_1,x_2)&\dots&k(x_1,x_m)\\
    \vdots&\:&\:\vdots\:\\
    k(x_m,x_1)&k(x_m,x_2)&\dots&k(x_m,x_m)
    \end{pmatrix},\quad \mu_A=\begin{pmatrix}
    \mu(x_1)\\\vdots\\\mu(x_m)
    \end{pmatrix}$
    $k$ is called **covariance (kernel)** function
    $\mu$ is called **mean** function
* **GP Marginals**
Typically, primarily interested in marginals, i.e.,
$p(f(x))=\mathcal{N}(f(x);\mu(x),k(x,x))$
$k(x_1,x_2)=Cov(f(x_1),f(x_2))=\mathbb{E}[((f(x_1)-\mu(x_1))((f(x_2)-\mu(x_2))]$
$k(x,x)=Cov(f(x),f(x))=\mathbb{E}[((f(x)-\mu(x))^2]=Var(f(x))$

**Covariance (kernel) Functions**
* $k$ must be **symmetric**
    * $k(x,x')=k(x',x) \text{ for all }x,x'$
* $k$ must be **positive definite**
    * For all $A$: $K_{AA}$ is positive definite matrix
    * $\forall x \in \mathbb{R}^{|A|}: x^TK_{AA}x\geq 0 \Leftrightarrow \text{all eigenvalues of } K_{AA}\geq 0$
* Kernel function $k$: assumptions about correlation!

**Covariance Functions: Examples**
* Linear kernel:$k(x,x')=x^Tx'$
    * GP with linear kernel = Bayesian linear regression
    * Linear kernel with features:
    $k(x,x')=\phi(x)^T\phi(x')$
* Squared exponential (a.k.a. RBF, Gaussian) kernel
    * $k(x,x')=exp(-\|x-x'\|_2^2/h^2)$ , $h$ is called bandwidth
* Exponential kernel
    * $k(x,x')=exp(-\|x-x'\|_2/h)$

**Smoothness of GP Samples**
* Covariance function determines smoothness of sample paths*
assuming $\mu(x)=0, \forall x$
    * Squared exponential kernel: **analytic** (**infinitely** diff'able)
    * Exponential kernel: continuous, but **nowhere** diff’able
    * Matérn kernel with parameter $\nu$: $\lceil\nu\rceil$ times (m.s.) diff’able
    $k(\mathbf{x},\mathbf{x}')=\frac{2^{1-\nu}}{\Gamma(\nu)}(\frac{\sqrt{2\nu}\|\mathbf{x}-\mathbf{x}'\|_2}{\rho})^\nu K_\nu(\frac{\sqrt{2\nu}\|\mathbf{x}-\mathbf{x}'\|_2}{\rho})$
    Hereby $\Gamma$ is the Gamma function, $K_\nu$ the modified Bessel function of the second kind, and $\rho$ is a bandwidth parameter.
    * Special cases: $\nu=\frac{1}{2}$ gives exponential kernel; $\nu \rightarrow \infin$ gives Gaussian kernel.

**Composition Rules**
* Suppose we have two covariance functions.
$k_1:\mathcal{X}\times\mathcal{X} \rightarrow \mathbb{R} \quad k_2:\mathcal{X}\times\mathcal{X} \rightarrow \mathbb{R}\qquad$definedon data space $\mathcal{X}$
* Then the following functions are valid cov. functions:
$k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')+k_2(\mathbf{x},\mathbf{x}') $
$\qquad \rightarrow f_1\sim GP(\mu_1,k_1),f_2\sim GP(\mu_2,k_2),g=f_1+f_2\sim GP(\mu_1+\mu_2,k_1+k_2)$
$k(\mathbf{x},\mathbf{x}')=k_1(\mathbf{x},\mathbf{x}')k_2(\mathbf{x},\mathbf{x}')$
$k(\mathbf{x},\mathbf{x}')=c\:k_1(\mathbf{x},\mathbf{x}')\quad \text{for }c>0$
$k(\mathbf{x},\mathbf{x}')=f(k_1(\mathbf{x},\mathbf{x}'))$, where $f$ is a  polynomial with positive coefficients or the exponential function

**Forms of Covariance Functions**
* Covariance function $k:\mathbb{R}^d\times \mathbb{R}^d \rightarrow \mathbb{R}$ is called:
    * **Stationary** if $k(x,x')=k(x-x')$
    * **Isotropic** if $k(x,x')=k(\|x-x'\|_2)$
    $\begin{matrix}
    \: & Stationary?&Isotropic?\\
    Linear& \times& \times\\
    Gaussian& \checkmark& \checkmark\\
    exp(-\frac{(x-x')^T\mathbf{M}(x-x')}{h^2}) & \checkmark & \times\\
    \mathbf{M} \:pos.semi-def.&\:&\:
    \end{matrix}$

**Making Predictions with GPs**
* Suppose $p(f)=GP(f;\mu;k)$
and we observe $y_i)=f(\mathbf{x}_i +\varepsilon_i)\quad \varepsilon_i\sim\mathcal{N}(0,\sigma^2)\quad A=\{\mathbf{x}_1,...,\mathbf{x}_m\}$
* Then $p(f|\mathbf{x}_1,...,\mathbf{x}_m,y_1,...,y_m)=GP(f;\mu',k)'$
where 
$\mu'(\mathbf{x})=\mu(\mathbf{x})+\mathbf{k}_{x,A}(\mathbf{K}_{AA}+\sigma^2\mathbf{I})^{-1}(\mathbf{y}_A-\mu_A)$
$k'(\mathbf{x},\mathbf{x}')=k(\mathbf{x},\mathbf{x}')-\mathbf{k}_{x,A}(\mathbf{K}_{AA}+\sigma^2\mathbf{I})^{-1}\mathbf{k}_{x',A}$
*Note*: $\mathbf{k}_{x,A}=[k(x,x_1),...k(x,x_m)]$
* $\rightarrow$ Closed form formulas for prediction!
* $\rightarrow$ Posterior covariance $k'$ does not depend on $\mathbf{y}_A$

**Common Convention: Prior Mean 0**
Surpose $f\sim GP(\mu,k)$
Define $g:=g(x)=f(x)-\mu(x)\quad \forall x$
$\Rightarrow g\sim GP(0,k)$
$\Rightarrow f(x)=g(x)+\mu(x)$

**How to sample from a GP?**
* Forward sampling
$P(f_1,...,f_n)=P(f_1)P(f_2|f_1)...P(f_n|f_{1:n-1})$
where $P(f_1)\sim\mathcal{N}(\mu_1,\sigma_1^2),...,P(f_n|f_{1:n-1})\sim\mathcal{N}(\mu_{n|1:n-1},\sigma_{n|1:n-1}^2)$
Can sample $f_1\sim P(f_1)$ ,Then $f_2\sim P(f_2|f_1)...f_n\sim P(f_n|f_{1:n-1})$

**Side Note: Kalman Filters are GPs**
* **Kalman filters** can be seen as **a special case of a GP** with a particular conditional independence structure that allows efficient / recursive Bayesian filtering
* $\{x_1,x_2...y_1,y_2,...\}$ is a GP, $x_1\sim\mathcal{N}(0,\sigma_p^2)$
$x_{t+1}=x_t+\varepsilon_t,\quad \varepsilon_t\sim\mathcal{N}(0,\sigma_x^2)$
$y_t=x_t+\eta_t,\quad \eta_t\sim\mathcal{N}(0,\sigma_y^2)$
*Note*:
$\sigma_1^2=\sigma_p^2,\quad \sigma_2^2=\sigma_p^2+\sigma_x^2, \quad \sigma_t^2=\sigma_p^2+(t-1)\sigma_x^2$
$\mu_{t+1}=\mathbb{E}[x_{t+1}]=\mathbb{E}[x_t+\varepsilon_t]=\mu_t+\mathbb{E}[\varepsilon_t]=\mu_t=\mu_1=0$
$Cov(x_t,x_{t+\Delta})=\mathbb{E}[(x_t-\mu_t)(x_{t+\Delta}-\mu_{t+\Delta})]=...=Var(x_t^2)=\sigma_t^2$


**Optimizing Kernel Parameters**
* How should we pick the hyperparameters?
* One answer: crossvalidation on predictive performance.
* The Bayesian perspective provides an alternative approach: 
**Maximize the marginal likelihood of the data**
* $\hat{\theta}=argmax_\theta p(y|x,\theta)=argmax_\theta \int p(y,f|x,\theta)df$
$=argmax_\theta \int p(y|f,x)p(f|\theta)df=argmax_\theta \mathcal{N}(y;0,\mathbf{K}_y(\theta))\leftarrow $zero mean by convention
$=argmin_\theta \frac{d}{2}log2\pi+\frac{1}{2}log|\mathbf{K}_y(\theta)|+\frac{1}{2}y^T\mathbf{K}_y(\theta)y$
*Note*: $\theta=[\theta',\sigma_n^2], \quad \mathbf{K}_y(\theta)=\mathbf{K}_x(\theta')+\sigma_n^2\mathbf{I}$
$\mathcal{N}(y;0,\mathbf{K}_y(\theta))=\frac{1}{\sqrt{(2\pi)^d|\mathbf{K}_y(\theta)|}}exp(-\frac{1}{2}y^T\mathbf{K}_y(\theta)y)$

**Model Selection for GPs**
* Marginal likelihood of the data
$logp(\mathbf{y}|X,\theta)=-\frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y}-\frac{1}{2}log|\mathbf{K}_y|-\frac{n}{2}log2\pi \quad$ the last term is indep. of $\theta$
* Can find $\hat{\theta}=argmax\:p(\mathbf{y}|X,\theta)$ by gradient descent
$\hat{\theta}=argmin_\theta \: \frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y}+\frac{1}{2}log|\mathbf{K}_y|=argmin_\theta \: \mathcal{L}(\theta)$

**Optimizing the Likelihood**
* Gradient of the Likelihood
$\frac{\partial}{\partial\theta_j}log\: p(\mathbf{y}|X,\theta)=\frac{1}{2}\mathbf{tr}((\alpha\alpha^T-\mathbf{K}^{-1})\frac{\partial\mathbf{K}}{\partial\theta_j}$ , where $\alpha=\mathbf{K}^{-1}\mathbf{y}$
* probably converge to local optima

**Bayesian Model Melection**
* $p(y|X,\theta)=\int p(y|f,X)p(f|\theta)df$
* $\begin{matrix}
\: & p(y|f,X) & p(f|\theta)\\
\text{underfit(too simple)} & \text{small for most }f & \text{large}\\
\text{overfit(too complex)} & \text{large for few }f \text{,small for most }f& \text{small}\\
\text{just right} & \text{moderate} & \text{moderate}\\
\end{matrix}$
* In contrast, MAP estimation approx. $p(y|X,\theta)\approx p(y|\hat{f},\theta)$
where $\hat{f}=argmax \: p(y|f,X)p(f|\theta)$
* **Maximizing marginal likelihood** is an example of an **Empirical Bayes method** – estimating a prior distribution from data
* Integrating (rather than optimizing) over the unknown function **helps guarding against overfitting**
* Other possibilities exist:
    * Can place **hyperprior** on parameters of the prior and obtain MAP estimate (corresponds to a regularization term)
    * Can integrate out the hyperprior (but also has params...)
    Instead of $\hat{\theta}=argmax_\theta \: p(y|X,\theta)$, can place hyperprior $p(\theta)$ on $\theta$
    $\rightarrow \hat{\theta}=argmax_\theta \: p(\theta|X,y)$
    $=argmax_\theta \: p(\theta)p(y|X,\theta)=argmin_\theta \: -logp(y|X,\theta)-logp(\theta)$
    * Or go **fully bayesian**
    $p(y^*|x^*,X,y)=\int p(y^*|x^*,f)p(f|X,y,\theta)p(\theta)dfd\theta$

**Computational Issues**
* Computational cost of prediction with a GP?
$\mu'(\mathbf{x})=\mu(\mathbf{x})+\mathbf{k}_{x,A}(\mathbf{K}_{AA}+\sigma^2\mathbf{I})^{-1}(\mathbf{y}_A-\mu_A)$
$k'(\mathbf{x},\mathbf{x}')=k(\mathbf{x},\mathbf{x}')-\mathbf{k}_{x,A}(\mathbf{K}_{AA}+\sigma^2\mathbf{I})^{-1}\mathbf{k}_{x',A}^T$
* $\rightarrow$Exact computation requires solving linear system $(\mathbf{K}_{AA}+\sigma^2\mathbf{I})\cdot Z\quad $ in  $|A|=n\:$variables
* $\rightarrow \: \varTheta(|A|^3)$
* This is in contrast to Bayesian linear regression: $\varTheta(nd^2)$ (can even be maintained recursively at same cost)

**Fast GP Methods**
* Basic approaches for acceleration:
    * Exploiting parallelism (GPU computations)
    * Local GP methods
    * Kernel function approximations (RFFs, QFFs,...)
    * Inducing point methods (SoR, FITC, VFE etc.)

**Fast GPs: Exploiting parallelism**
* GP inference requires solving linear systems
* Resulting algorithms can be implemented on multicore (GPU) hardware
* Implemented by modern GP libraries (e.g., GPflow, GPyTorch)
* Yields substantial speedup, but doesn’t address the cubic scaling in $n$

**Fast GPs: Local Methods**
* Covariance functions that decay with distance of points (e.g., RBF, Matern, kernels) lend themselves to local computations
* To make a prediction at point $x$, only condition on points $x’$ where $|Cov(x,x’)|>\mathbf{\tau}$
for RBF kernel, this is equivalent to $\|x-x'\|<\tau'$
* Still expensive if “many” points close by

**Fast GPs: Kernel Function Approximation**
* Key idea: construct **explicit ''low-dimensional'' feature map** that approximates the true kernel function
$k(x,x')\approx \phi(x)^T\phi(x') \qquad \phi(x) \in \mathbb{R}^m$
* Then apply Bayesian linear regression
$\rightarrow$Computational cost:$O(nm^2+m^3)$ instead of $O(n^3)$
*  Different variations of this idea: Random Fourier Features, Nystrom Features,...

**Shift-invariant Kernels**
* A kernel $k(\mathbf{x},\mathbf{y})\quad \mathbf{x},\mathbf{y} \in \mathbb{R}^d$
is called **shift-invariant** if $k(\mathbf{x},\mathbf{y})=k(\mathbf{x}-\mathbf{y})$
* Such a kernel has a **Fourier transform**:
$k(\mathbf{x}-\mathbf{y})=\int_{\mathbb{R}^d}p(\omega)e^{j\omega^T(\mathbf{x}-\mathbf{y})}d\omega$
E.g. Gaussian Kernel $k(\mathbf{x},\mathbf{y})=exp(-\|\mathbf{x}-\mathbf{y}\|_2^2/2)$
has the Fourier Transform:
$p(\omega)=(2\pi)^{-d/2}exp(-\|\omega\|_2^2/2)$
This is simply the standard Gaussian distribution in D dimensions!
* Theorem [Bochner]: A shift-invariant kernel is **positive definite** if and only if $p(\omega)$ is **nonegative**
* Can scale the data, so that  $p(\omega)$ is a **probability distr.**!

**Random Fourier Features**
* Key idea: Interpret kernel as **expectation**
$k(\mathbf{x}-\mathbf{y})=\int_{\mathbb{R}^d}p(\omega)e^{j\omega^T(\mathbf{x}-\mathbf{y})}d\omega=\mathbb{E}_{\omega,b}[\mathbf{z}_{\omega,b}(\mathbf{x})\: \mathbf{z}_{\omega,b}(\mathbf{y})]$
$\approx \frac{1}{m}\sum_{i=1}^m \mathbf{z}_{\omega^i,b^i}(\mathbf{x}) \: \mathbf{z}_{\omega^i,b^i}(\mathbf{y})=\phi(\mathbf{x})^T\phi(\mathbf{y})$
where $\omega\sim p(\omega)$, $b\sim U([0,2\pi])$ , $\mathbf{z}_{\omega,b}(\mathbf{x})=\sqrt{2}cos(\omega^T\mathbf{x}+b)$
and $\phi(\mathbf{x})=\frac{1}{\sqrt{m}}(\mathbf{z}_{\omega^1,b^1}(\mathbf{x}),...,\mathbf{z}_{\omega^m,b^m}(\mathbf{x}))$
*  [RR NIPS‘07]
$\begin{matrix}
\text{Kernel Name} & k(\Delta) & p(\omega)\\
\text{Gaussian} & e^{-\frac{\|\Delta\|_2^2}{2}} & (2\pi)^{-\frac{D}{2}}e^{-\frac{\|\omega\|_2^2}{2}}\\
\text{Laplacian} & e^{-\|\Delta\|_1} & \prod_d \frac{1}{\pi(1+\omega_d^2)}\\
\text{Cauchy} & \prod_d \frac{2}{1+\Delta_d^2} & e^{-\|\Delta\|_1}\\
\end{matrix}$
* Performance of random features: 
* Bayesian linear regression with explicit feature map $z$ approximates GP

**Fourier Features can be wasteful**
* Fourier features approximate the kernel function **uniformly well**:
$Pr[\sup_{x,y\in \mathcal{M}} \: \|\mathbf{z}(\mathbf{x})'\mathbf{z}(\mathbf{y})-k(\mathbf{x},\mathbf{y})\|\geq \epsilon]\leq 2^8 (\frac{\sigma_p diam(\mathcal{M})}{\epsilon})^2exp(-\frac{D\epsilon^2}{4(d+2)})$
* This may be ''too much to ask'' : Only need accurate representation for training (and test) points!

**Inducing Point Methods**
* ''Summarize'' data via function values of $f$ at a set $\mathbf{u}$ of $m$ inducing points
$p(\mathbf{f}^*,\mathbf{f})=\int p(\mathbf{f}^*,\mathbf{f},\mathbf{u})d\mathbf{u}=\int p(\mathbf{f}^*,\mathbf{f}|\mathbf{u})p(\mathbf{u})d\mathbf{u}$
* Key idea: Approximate by
$p(\mathbf{f}^*,\mathbf{f})\approx q(\mathbf{f}^*,\mathbf{f})=\int q(\mathbf{f}^*|\mathbf{u})q(\mathbf{f}|\mathbf{u})p(\mathbf{u})d\mathbf{u}$
* Hereby, $q(\mathbf{f}^*|\mathbf{u})$ and $q(\mathbf{f}|\mathbf{u})$ are approximations of 
*Training* conditional $p(\mathbf{f}|\mathbf{u})=\mathcal{N}(\mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u},\mathbf{K}_{\mathbf{f},\mathbf{f}}-\mathbf{Q}_{\mathbf{f},\mathbf{f}})$
*Testing conditional* $p(\mathbf{f}^*|\mathbf{u})=\mathcal{N}(\mathbf{K}_{\mathbf{f}^*,\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u},\mathbf{K}_{\mathbf{f}^*,\mathbf{f}^*}-\mathbf{Q}_{\mathbf{f}^*,\mathbf{f}^*})$
where $\mathbf{Q}_{\mathbf{a},\mathbf{b}}\equiv \mathbf{K}_{\mathbf{a},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{K}_{\mathbf{u},\mathbf{b}}$

**Example: Subset of Regressors (SoR)**
* The Subset of Regressors (SoR) approximation replaces
$p(\mathbf{f}|\mathbf{u})=\mathcal{N}(\mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u},\mathbf{K}_{\mathbf{f},\mathbf{f}}-\mathbf{Q}_{\mathbf{f},\mathbf{f}})$
* By
$p(\mathbf{f}|\mathbf{u})=\mathcal{N}(\mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u},0)$
* Can show: the resulting model is a degenerate GP with covariance function
$k_{SoR}(\mathbf{x},\mathbf{x}')=k(\mathbf{x},\mathbf{u})\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}k(\mathbf{u},\mathbf{x}')$

**Example: Fully Independent Training Conditional (FITC)**
* The FITC approximation replaces
$p(\mathbf{f}|\mathbf{u})=\mathcal{N}(\mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u},\mathbf{K}_{\mathbf{f},\mathbf{f}}-\mathbf{Q}_{\mathbf{f},\mathbf{f}})$
* By
$q_{FITC}(\mathbf{f}|\mathbf{u})=\prod_{i=1}^n p(f_i|\mathbf{u})=\mathcal{N}(\mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u},diag(\mathbf{K}_{\mathbf{f},\mathbf{f}}-\mathbf{Q}_{\mathbf{f},\mathbf{f}}))$

* Computational Cost
* The computational cost for inducing point methods SoR and FITC is dominated by the cost of inverting $\mathbf{K}_{\mathbf{u},\mathbf{u}}$
* Thus, it is cubic in the number of inducing points, but linear in the number of data points

**How to Pick Inducing Points?**
* **Subsets of training data**?
    * Chosen randomly
    * Chosen greedily according to some criterion (e.g., variance)
* **Equally spaced in the domain**?
    * Random points
    * Deterministic grid
* **Optimized**?
    * Can treat $\mathbf{u}$ as hyperparameters and maximize marginal likelihood of the data
* Need to ensure $\mathbf{u}$ is representative of the data and where predictions are made

**Summary**
* **Gaussian processes = kernelized Bayesian Linear Regression**
* Can compute marginals / conditionals in **closed form**
* Optimize hyperparameters via **maximizing the marginal likelihood**
* Kalman filters are a **special case** of Gaussian processes


## Approximate Inference
### Lecture Notes
**Bayesian learning more generally**
* Prior: $p(\theta)$
* Likelihood: $p(y_{1:n}|x_{1:n},\theta)\prod_{i=1}^n p(y_i|x_i,\theta)$
* Posterior: $p(\theta|x_{1:n},y_{1:n})=\frac{1}{Z}p(\theta)\prod_{i=1}^n p(y_i|x_i,\theta)$
where $Z=\int p(\theta)\prod_{i=1}^np(y_i|x_i,\theta)d\theta$
* Predictions: $p(y^*|x^*,x_{1:n},y_{1:n})=\int p(y^*|x^*,\theta)p(\theta|x_{1:n},y_{1:n})d\theta$
* For Bayesian linear regression and GP regression, these(high-dimensional) integrals are closed-form! :smile:
* In general, this is not the case $\rightarrow$ need approximations
    * Example: Bayesian logistic regression
    $y\in \{1,-1\}$
    $\sigma(\mathbf{w}^T\mathbf{x})=\frac{1}{1+exp(-\mathbf{w}^T\mathbf{x})}$
    $p(y|\mathbf{x},\mathbf{w})=Ber(y;\sigma(\mathbf{w}^T\mathbf{x}))=\sigma(y \cdot \mathbf{w}^T\mathbf{x})$
    $p(\mathbf{w})=\mathcal{N}(0,\sigma_p^2\mathbf{I})$
    $p(y_{1:n}|x_{1:n},\mathbf{w})=\prod_{i=1}^n p(y_i|x_i,\mathbf{w})=\prod_{i=1}^n\sigma(y;\mathbf{w}^T\mathbf{x})$

**Approximate Inference**
* Will discuss general approaches for performing approximate inference in intractable distributions (i.e., partition function / normalizer hard to compute)
$p(\theta|y)=\frac{1}{Z}p(\theta,y)$
* Hereby, y are the observations (the data), and $\theta$ the latent variables (the parameters)
* We’ll assume we can evaluate the joint distribution, but not the normalizer $Z$
* Note that we often leave out the inputs $\mathbf{x}$ to keep notation simple
$p(y|\theta)\equiv p(y|\theta,x)$

**General Approaches**
* **Variational inference** seeks to approximate the intractable distribution $p$ by a simple one $q$ that is ''as close as possible''
$p(\theta|y)=\frac{1}{Z}p(\theta,y)\approx q(\theta|\lambda)$
* **Markov-Chain Monte Carlo** methods seek to approximate $p$ by (approximate) samples from $p$ (constructed by simulating a Markov Chain)

**Laplace Approximation**
* Laplace approximation uses a Gaussian approximation to the posterior distribution obtained from a second-order Taylor expansion around the **posterior mode**
* $q(\theta)=\mathcal{N}(\theta;\hat{\theta},\Lambda^{-1})$
$\hat{\theta}=argmax_\theta p(\theta|y)$
$\Lambda=-\nabla\nabla logp(\hat{\theta}|y)$
* *Note*: $f(\theta)\equiv logp(\theta|y)\quad f(\theta)\approx f(\hat{\theta})+\nabla f_{\hat{\theta}}(\theta-\hat{\theta})+\frac{1}{2}(\theta-\hat{\theta})^T[\nabla\nabla f_{\hat{\theta}}](\theta-\hat{\theta})$
any $p$ s.t. $logp(x)=c-x^T\Lambda x$ must be Gaussian

**Laplace Approx. for Bayesian log. regression**
* $p(w)=\mathcal{N}(w;0,\sigma_p^2\mathbf{I})=\frac{1}{Z'}exp(-\frac{1}{2\sigma_p^2}\|w\|_2^2);\quad p(y_{1:n}|w)=\prod_{i=1}^n \sigma(y_i;w^Tx_i)$
$\hat{w}=argmax_w p(w|y_{1:n})=argmax_w \frac{1}{Z}p(w)p(y_{1:n}|w)$
$=argmax_w logp(w)+logp(y_{1:n}|w)$
$=argmax_w -logZ'-\frac{1}{2\sigma_p^2}\|w\|_2^2+\sum_{i=1}^nlog \sigma(y_i;w^Tx_i) $
$=argmin_w \frac{1}{2\sigma_p^2}\|w\|_2^2+\sum_{i=1}^nlog (1+exp(-y_iw^Tx_i))$
* *Note*: $\sigma(z)=\frac{1}{1+exp(-z)}\quad log\sigma(z)=-log(1+exp(-z))$
$\lambda=\frac{1}{2\sigma_p^2}$

**Finding the Mode**
* $\hat{w}=argmax_w p(w|y_{1:n})=argmin_w \sum_{i=1}^nlog (1+exp(-y_iw^T x_i))+\lambda\|w\|_2^2$
* This is just **standard (regularized) logistic regression**!
* Can solve, e.g., using stochastic gradient descent (see introduction to ML)
* Don’t need to know normalizer $Z$

**Recall: Stochastic Gradient Descent**
* Goal: minimize stochastic objectives
$L(\theta):=\mathbb{E}_{\mathbf{x}\sim p}l(\theta;\mathbf{x})$
* SGD:
    * Initialize $\theta_1$
    * For $t=1$ to $T$
        * Draw minibatch $B=\{\mathbf{x}_1,...,\mathbf{x}_m\},\mathbf{x}_i\sim p$
        * Update $\theta_{t+1}\leftarrow \theta_t-\eta_t\frac{1}{m}\sum_{i=1}^m\nabla_\theta l(\theta_t;\mathbf{x}_i)$
* Many variants (Momentum, AdaGrad, ADAM,...)
* For proper learning rate converges to (local) minimum
* Gradient $\nabla_\theta l(\theta_t;\mathbf{x}_i)$ often obtained by automatic differentiation
* One way to choose learning rate: $\sum_t \eta_t=\infin,\quad \sum_t \eta_t^2<\infin\qquad$E.g. $\eta_t=\frac{c}{t}$

**Recall: SGD for Logistic Regression**
* Initialize $\mathbf{w}$
* For $t=1,2,...$
    * Pick data point $(\mathbf{x},y)$ uniformly at random from data $D$
    * Compute probability of misclassification with current model 
    $\hat{P}(Y=-y|\mathbf{w},\mathbf{x})=\frac{1}{1+exp(y\mathbf{w}^T\mathbf{x})}$
    * Take gradient step
    $\mathbf{w}\rightarrow \mathbf{w}(1-2\lambda\eta_t)+\eta_ty\mathbf{x}\hat{P}(Y=-y|\mathbf{w},\mathbf{x})$

**Finding the Covariance**
* $\Lambda=-\nabla\nabla log p(\hat{\mathbf{w}}|\mathbf{x}_{1:n},y_{1:n})=\sum_{i=1}^n \mathbf{x}_i\mathbf{x}_i^t\pi_i(1-\pi_i)=\mathbf{X}^Tdiag([\pi_i(1-\pi_i)]_i)\mathbf{X}$
* where $\pi_i=\sigma(\hat{\mathbf{w}}^T\mathbf{x}_i)$
* *Note*: $\nabla\nabla log\frac{1}{Z}p(\theta,y)=\nabla(\nabla log\frac{1}{Z}+\nabla logp(\theta,y))=\nabla\nabla logp(\theta,y)$
* Crucially, $\Lambda$ does not depend on the normalizer $Z$

**Making Predictions**
* Suppose want to predict 
$p(p^*|x^*,x_{1:n},y_{1:n})=\int p(y^*|x^*,w)p(w|x_{1:n},y_{1:n})dw\approx \int p(y^*|x^*,w)q_\lambda(w)dw$
$=\int p(y^*|f^*)q(f^*)df^*$
This integral still has no closed form, but is easy to approximate(to machine precision), e.g. Gauss-Hermite quadrature $f(x)\approx \sum_i w_if(x_i)$
Can also do sample based approx: $w^{(1)},...,w^{(m)}\sim q_\lambda,\quad p(y^*|...)=\frac{1}{m}\sum_{i=1}^mp(y^*|x^*,w^{(i)})$
* *Note*:
    * $f^*=w^Tx^*,\quad p(y^*|f^*)=\sigma(y^*f^*)$ 
    * $q(f^*)\equiv \int p(f^*|w)q_\lambda(w)dw$
    If $q_\lambda=\mathcal{N}(\hat{w},\Lambda^{-1})\rightarrow q(f^*)=\mathcal{N}(f^*;\hat{w}^Tx^*,{x^*}^T\Lambda^{-1} x^*)$
* This one-dimensional integral can be easily approximated efficiently using numerical quadrature
* [Side note: For other link functions (e.g., Gaussian CDF), can even be calculated analytically]

**Issues with Laplace Approximation**
* Laplace approximation first greedily seeks the mode, and then matches the curvature
* his can lead to poor (e.g., overconfident) approximations 

**Variational Inference**
* Given unnormalized distribution
$p(\theta|y)=\frac{1}{Z}p(\theta,y)$
* Try to find a “simple” (tractable) distribution that approximates p well
$q^*\in argmin_{q\in \mathcal{Q}} \: KL(q\|p)=argmin_{\lambda\in \mathbb{R}^D} \: KL(q_\lambda\|p)$

**Simple Distributions**
* Need to specify a **variational family** (of simple distributions)
* E.g.: Gaussian distributions; Gaussians with diagonal covariance,...
$\mathcal{Q}=\{q(\theta)=\mathcal{N}(\theta;\mu,diag([\sigma]))\}$
$q=q_\lambda$, where $\lambda=[\mu,\sigma^2]$

**KL-Divergence**
* Given distributions $q$ and $p$, Kullback-Leibler divergence between $q$ and $p$ is
$KL(q\|p)=\int q(\theta)log\frac{q(\theta)}{p(\theta)}d\theta=\mathbb{E}_{\theta\sim q}[log\frac{q(\theta}{p(\theta}]$
Typically, we assume $p\&q$ have same support
* Properties
    * Non-negative: $KL(q\|p)\geq 0\quad \forall q,p$
    * Zero if and only if $p\&q$ agree almost everywhere: $KL(q\|p)=0\Leftrightarrow q=p$
    * Not generally symmetric: $KL(q\|p)\neq KL(p\|q)$

**Example: KL Divergence Between Gaussians**
* Consider two Gaussian distributions $p$ and $q$
$p=\mathcal{N}(\mu_0,\Sigma_0), \: q=\mathcal{N}(\mu_1,\Sigma_1)$
* Then it holds that
$KL(p\|q)=\frac{1}{2}(tr(\Sigma_1^{-1}\Sigma_0)+(\mu_1-\mu_0)^T\Sigma_1^{-1}(\mu_1-\mu_0)-d+ln(\frac{|\Sigma_1|}{|\Sigma_0|}))$
* If $p=\mathcal{N}([\mu_1,...,\mu_d],diag([\sigma_1^2,...,\sigma_d^2]))$ and $q=\mathcal{N}(0,I)$
$KL(p\|q)=\frac{1}{2}\sum_{i=1}^d(\sigma_i^2+\mu_i^2-1-ln\sigma_i^2)$
* Suppose $p=\mathcal{N}(\mu_0,I),q=\mathcal{N}(\mu_1,I)$
$KL(p\|q)=\frac{1}{2}\|\mu_0-\mu_1\|_2^2$

**Entropy**
* Entropy of a distribution:
$H(q)=-\int q(\theta)logq(\theta)d\theta=\mathbb{E}_{\theta\sim q}[-logq(\theta)]$
* Entropy of a product distribution: $q(\theta_{1:d})=\prod_{i=1}^d q_i(\theta_i)$
$H(q)=\sum_{i=1}^d H(q_i)$
* Example: Entropy of a Gaussian
$H(\mathcal{N}(\mu,\Sigma))=\frac{1}{2}ln|2\pi e\Sigma|$
For $\Sigma=diag(\sigma_1^2,...,\sigma_d^2)\Rightarrow H=\frac{1}{2}ln|2\pi e| +\sum_{i=1}^d ln\sigma_i^2$

**Minimizing KL Divergence**
* $\begin{aligned}argmin_q KL(q\|p)&=argmin_q\int q(\theta)log\frac{q(\theta)}{\frac{1}{Z}p(\theta,y)}d\theta\\
&=argmax_q \int q(\theta)[logp(\theta,y)-logZ-logq(\theta)]d\theta\\
&=argmax_q\int q(\theta)logp(\theta,y)d\theta+H(q)\\
&=argmax_q\mathbb{E}_{\theta\sim q(\theta)}[logp(\theta,y)]+H(q)\\
&=argmax_q\mathbb{E}_{\theta\sim q(\theta)}[logp(y|\theta)]-KL(q\|p(\cdot))\end{aligned}$
* *Note*:
$p$ is posterior, $p(\cdot)$ is prior

**Maximizing Lower Bound on Evidence**
* $\begin{aligned}
logp(y)&=log\int p(y|\theta)p(\theta)d\theta\\
&=log\int p(y|\theta)\frac{p(\theta)}{q(\theta)}q(\theta)d\theta\\
&=log\mathbb{E}_{\theta\sim q}[p(y|\theta)\frac{p(\theta)}{q(\theta)}]\\
&\geq \mathbb{E}_{\theta\sim q}[log(p(y|\theta)\frac{p(\theta)}{q(\theta)})]\\
&=\mathbb{E}_{\theta\sim q}[log(p(y|\theta)]d\theta-KL(q\|p(\cdot))\end{aligned}$

**Inference as Optimization**
* Thus,
$\begin{aligned}argmin_q KL(q\|p(\cdot|y))&=argmax_q\mathbb{E}_{\theta\sim q(\theta)}[logp(y|\theta)]-KL(q\|p(\cdot))\\
&=argmax_q\mathbb{E}_{\theta\sim q(\theta)}[logp(\theta,y)]+H(q)\\
&=argmax_qL(q)
\end{aligned}$
* Thus, prefer distributions q that maximize the expected (**joint/conditional**) data likelihood, but are also **uncertain / close** to the prior
* *Note*:
$L(q)$ is called **''ELBO'' (Evidence lower bound)**
$L(q)\leq log(p(y)\leftarrow $evidence

**ELBO for Bayesian Logistic Regression**
* $L(\lambda)=\mathbb{E}_{\theta\sim q(\cdot|\lambda)}[logp(y|\theta)]-KL(q_\lambda\|p(\cdot))$
Suppose: $Q$ is diagonal Gaussians $\rightarrow$ $\lambda=[\mu_{1:d},\sigma_{1:d}^2]\in \mathbb{R}^{2d},\quad p(\theta)=\mathcal{N}(0,I)$
$\rightarrow KL(q_\lambda\|p(\cdot))=\frac{1}{2}\sum_{i=1}^d(\mu_i^2+\sigma_i^2-1-ln\sigma_i^2)$
* $\begin{aligned}\mathbb{E}_{\theta\sim q_\lambda}[log(p(y|\theta)]&=\mathbb{E}_{\theta\sim q_\lambda}[\sum_{i=1}^nlog(p(y_i|\theta,x_i)]\\
&=\mathbb{E}_{\theta\sim q_\lambda}[-\sum_{i=1}^nlog(1+exp(-y_i\theta^Tx_i))]\end{aligned}$

**Gradient of the ELBO**
* $\begin{aligned}
\nabla_\lambda L(\lambda)&=\nabla_\lambda [\mathbb{E}_{\theta\sim q(\cdot|\lambda)}[logp(y|\theta)]-KL(q_\lambda\|p(\cdot))]\\
&=\nabla_\lambda [\mathbb{E}_{\theta\sim q(\cdot|\lambda)}[logp(\theta,y)]+H(q(\cdot|\lambda))]
\end{aligned}$
* Need to differentiate an expectation w.r.t. q
* Unfortunately **q depends on the variational params**.
* Key idea: Rewrite in a way that allows Monte Carlo approximation. Different approaches
    * Score gradients (not discussed further here)
    * Reparametrization gradients

**Reparameterization Trick**
* Suppose we have a random variable $\epsilon\sim\phi$ sampled from a base distribution, and consider $\theta=g(\epsilon,\lambda)$ for some invertible function $g$
* Then it holds that $q(\theta|\lambda)=\phi(\epsilon)|\nabla_\epsilon g(\epsilon;\lambda|^{-1}$
(change of variables for probability) and $\mathbb{E}_{\theta\sim q_\lambda}[f(\theta)]=\mathbb{E}_{\epsilon\sim\phi}[f(g(\epsilon ;\lambda))]$
* Thus, after reparameterization, the expectation is w.r.t. to distribution $\phi$ that **does not depend** on $\lambda$ !
* This allows to **obtain stochastic gradients** via
$\nabla_\lambda \mathbb{E}_{\theta\sim q_\lambda}[f(\theta)]=\mathbb{E}_{\epsilon\sim\phi}[\nabla_\lambda f(g(\epsilon ;\lambda))]$

**Example: Gaussians**
* Suppose we use a Gaussian variational approximation
$q(\theta|\lambda)=\mathcal{N}(\theta;\mu,\Sigma);\quad \lambda=[\mu,\Sigma]$
* Can reparametrize $\theta=g(\epsilon,\lambda)=C\epsilon+\mu$, such that $\Sigma=CC^T$ and $\phi(\epsilon)=\mathcal{N}(\epsilon;0,I)$
* Then it holds that $\epsilon=C^{-1}(\theta-\mu)$ and $\phi(\epsilon)=q(\theta|\lambda)|C|$
* Can w.l.o.g. choose $C$ to be positive definite and lower-diagonal($C$ is Cholesky factor of $\Sigma$)

**Reparametrizing the ELBO for Bayesian Logistic Regression**
* $\begin{aligned}
\nabla_\lambda L(\lambda)&=\nabla_\lambda [\mathbb{E}_{\theta\sim q(\cdot|\lambda)}[logp(y|\theta)]-KL(q_\lambda\|p(\cdot))]\\
&=\nabla_{C,\mu}\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[logp(y|C\epsilon+\mu)]-\nabla_{C,\mu}KL(q_{C,\mu}\|p(\cdot))
\end{aligned}$
* Can compute $\nabla_{C,\mu}KL(q_{C,\mu}\|p(\cdot))$ exactly (e.g., via automatic differentiation)
* Can obtain unbiased stochastic gradient estimate of 
$\begin{aligned}
&\nabla_{C,\mu}\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[logp(y|C\epsilon+\mu)]\\
=&\nabla_{C,\mu}\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[n\cdot \frac{1}{n}\sum_{i=1}^n logp(y_i|C\epsilon+\mu,x_i)]\\
=&\nabla_{C,\mu}n\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\mathbb{E}_{i\sim Unif\{1:n\}}logp(y_i|C\epsilon+\mu,x_i)\\
\approx & \nabla_{C,\mu}n\cdot\frac{1}{m} \sum_{j=1}^m logp(y_{i_j}|C\epsilon^{(j)}+\mu,x_{i_j})
\end{aligned}$
* *Note*:
    * Draw mini-batch $\epsilon^{(1)},...,\epsilon^{(m)}\sim \phi$
    * Draw $i_1,...,i_m\sim Unif\{1,...,n\}$

**Black Box Stochastic Variational Inference**
* Maximizing the ELBO using stochastic optimization (e.g., Stochastic Gradient Ascent)
* Can obtain unbiased gradient estimates, e.g., via **reparameterization trick**, or **score gradients**:
$\nabla_\lambda L(\lambda)=\mathbb{E}_{\theta\sim q_\lambda}[\nabla_\lambda logq(\theta|\lambda)(logp(y,\theta)-logq(\theta|\lambda))]$
* For diagonal $q$, only twice as expensive as MAP infer.
* Only need to be able to differentiate the (unnormalized) joint probability density $p$ and $q$
* Outlook: Can achieve better performance, e.g., using
    * Natural gradients
    * Variance reduction techniques (e.g., control variates) 

**Side Note: Gaussian Process Classification**
* All our discussions naturally generalize from Bayesian linear regression to Gaussian process classification:
$P(f)=GP(\mu,k)\quad P(y|f,\mathbf{x})=\sigma(y\cdot f(\mathbf{x}))$
* Often implemented using pseudo inputs, and maximizing the ELBO
$\sum_{i=1}^n\mathbb{E}_{q(f_i)}[logp(y_i|f_i)]-KL(q(\mathbf{u})\|p(\mathbf{u}))$
where $q(f_i):=\int p(f_i|\mathbf{u})q(\mathbf{u})d\mathbf{u}$

**Variational Inference Summary**
* Variational inference **reduces inference** (“summation/integration”) **to optimization**
* Can use highly efficient stochastic optimization techniques to find approximations
* Quality of approximation hard to analyze


## Markov Chain Monte Carlo
### Lecture Notes
**Approximating Predictive Distributions**
* Key challenge in Bayesian learning: Computing
$\begin{aligned}
p(y^*|x^*,x_{1:n},y_{1:n})&=\int p(y^*|x^*,\theta)p(\theta|x_{1:n},y_{1:n})d\theta\\
&=\mathbb{E}_{\theta\sim p(\cdot|x_{1:n},y_{1:n})}[f(\theta)]\\
&\approx \frac{1}{m}\sum_{i=1}^m f(\theta^{(i)})
\end{aligned}$
where $\theta^{(i)}\sim p(\theta|x_{1:n},y_{1:n})$
* If we had access to samples from the posterior, could use to obtain **Monte-Carlo approximation** of predictive distribution

**Sample Approximations of Expectations**
* $x_1,...x_N,...$ independent samples from $P(X)$
* (Strong) Law of large numbers:
$\mathbb{E}_P[f(X)]=lim_{N\to\infin}\frac{1}{N}\sum_{i=1}^Nf(x_i)$
* Hereby, the convergence is with probability 1
(almost sure convergence)
* Suggests **approximation** using **finite samples**:
$\mathbb{E}_P[f(X)]\approx\frac{1}{N}\sum_{i=1}^Nf(x_i)$

**How Many Samples Do We Need?**
* **Hoeffding’s inequality** Suppose $f$ is bounded in $[0,C]$. Then
$P(|\mathbb{E}_P[f(X)]-\frac{1}{N}\sum_{i=1}^Nf(x_i)|?\varepsilon)\leq 2exp(-2N\varepsilon^2/C^2)$
* Thus, probability of error decreases exponentially in N!

**Sampling From Intractable Distributions**
* Given unnormalized distribution
$P(x)=\frac{1}{Z}Q(x)$
* $Q(X)$ efficient to evaluate, but normalizer $Z$ intractable
* How can we sample from $P(X)$?
* Ingenious idea: Can create Markov chain that is efficient to simulate and that has stationary distribution $P(X)$

**Markov Chains**
* A (stationary) Markov chain is a sequence of RVs, $X_1,...,X_N,...$ with
    * Prior $P(X_1)
    * Transition probabilities $P(X_{t+1}|x_t)$ independent of $t$
    $X_{t+1}\bot X_{1:t-1}|X_t \quad \forall t$
    $P(X_{1:N})=P((X_1)P(X_2|X_1)...P(X_N|X_{N-1})$

**Ergodic Markov Chains**
* A Markov Chain is called ergodic, if there exists a finite $t$ such that every state can be reached from every state in exactly $t$ steps

**Stationary Distributions**
* An (stationary)ergodic Markov Chain has a unique and positive stationary distribution $\pi(X)>0$, s.t. for all x
$lim_{N\to \infin}P(X_N=x)=\pi(x)$
* The stationary distribution is independent of $P(X_1)$

**Simulating a Markov Chain**
* Can simulate a Markov chain via forward sampling:
$P(X_{1:N})=P((X_1)P(X_2|X_1)...P(X_N|X_{N-1})$
* If simulated “sufficiently long”, sample $X_N$ is drawn from a distribution “very close” to stationary distribution $\pi$

**Markov Chain Monte Carlo**
* Given an unnormalized distribution $Q(x)$
* Want to design a Markov chain with stationary distribution
$\pi(x)=\frac{1}{Z}Q(x)$
* Need to specify transition probabilities $P(x|x')$
* How can we choose them to ensure correct stationary distribution?

**Detailed Balance Equation**
* A Markov Chain satisfies the detailed balance equation for unnormalized distribution Q if for all $x, x’$:
$\frac{1}{Z}Q(x)P(x'|x)=\frac{1}{Z}Q(x')P(x|x')$
* Suffices to show: $P(X_t=x)=\frac{1}{Z}Q(x) \Rightarrow P(X_{t+1}=x)=\frac{1}{Z}Q(x)$
    * Assume $P(X_t=x)=\frac{1}{Z}Q(x)$
    * Then $\begin{aligned}
    P(X_{t+1}=x)&=\sum_{x'}P(X_{t+1}=x,X_t=x')\\
    &=\sum_{x'}P(X_{t+1}=x|X_t=x')P(X_t=x')\\
    &=\frac{1}{Z}\sum_{x'}P(x|x')Q(x')\\
    &\overset{D.B.}= \frac{1}{Z}\sum_{x'}P(x'|x)Q(x)\\
    &=\frac{1}{Z}Q(x)\sum_{x'}P(x'|x)\\
    &=\frac{1}{Z}Q(x)
    \end{aligned}$

**Designing Markov Chains**
* 1) Proposal distribution $R(X’|X)$
    * Given $X_t=x$, sample “proposal” $x’\sim R(X’|X=x)$
    * Note: Performance of algorithm will strongly depend on $R$
* 2) Acceptance distribution:
    * Suppose $X_t=x$
    * With probability $\alpha=min\{1,\frac{\frac{1}{Z}Q(x')R(x|x')}{\frac{1}{Z}Q(x)R(x'|x)}\}$ set $X_{t+1}=x’$
    * With probability $1-\alpha$, set $X_{t+1}=x$
* Theorem [Metropolis, Hastings]: The stationary distribution is $Z^{-1}Q(x)$
    * Proof: Markov chain satisfies detailed balance condition!

**MCMC for Random Vectors**
* Markov chain state can be a vector $\mathbf{X}=(X_1,...,X_n)$
* Need to specify proposal distributions $R(x’|x)$ over such random vectors
    * $x$: old state (joint configuration of all variables)
    * $x’$: proposed state, $x’\sim R(X’|X=x)$
* One popular example: Gibbs sampling!

**Gibbs Sampling: Random Order**
* Start with initial assignment $x$ to all variables
* Fix observed variables $X_B$ to their observed value $X_B$
* For $t=1$ to $\infin$ do 
    * Pick a variable $i$ uniformly at random from $\{1,...,n\}\backslash \mathbf{B}$
    * Set $\mathbf{v}_i=$values of all $x$ except $x_i$
    * Update $x_i$ by sampling from $P(X_i|\mathbf{v}_i)$
* Satisfies detailed balance equation!

**Gibbs Sampling: Practical Variant**
* Start with initial assignment $\mathbf{x}^{(0)}$ to all variables
* Fix observed variables $\mathbf{X}_\mathbf{B}$ to their observed value $\mathbf{x}_\mathbf{B}$
* For $t=1$ to $\infin$ do 
    * Set $\mathbf{x}^{(t)}=\mathbf{x}^{(t-1)}$
    * For each variable $X_i$(except those in $\mathbf{B}$)
        * set $\mathbf{v}_i$=values of all $\mathbf{x}^{(t)}$ except $x_i$
        * Sample ${x^{(t)}}_i$ from $P(X_i|\mathbf{v}_i)$ 
* No detailed balance, but also has correct stationary distribution.

**Computing $P(X_i|\mathbf{v}_i)$**
* Key insight in Gibbs sampling: Sampling from $X_i$ given an assignment to **all** other variables is (typ.) efficient!
* Generally, can compute 
$P(X_i|\mathbf{v}_i)=\frac{1}{Z}Q(X_i|\mathbf{v}_i)=\frac{1}{Z}Q(X_{1:N})$
where $Z=\sum_x Q(X_i=x,\mathbf{v}_i)$
* Thus, re-sampling $X_i$ only requires evaluating unnormalized joint distr. and renormalizing!
* Example: (Simple) Image Segmentation: see lecture notes

**Ergodic Theorem (special case)**
* Suppose $X_1,X_2,...,X_N,..$ is an ergodic Markov chain over a finite state space $D$, with stationary distribution $\pi$. Further let $f$ be a function on $D$.
* Then it holds a.s. that
$lim_{N\to\infin}\frac{1}{N}\sum_{i=1}^Nf(x_i)=\sum_{x\in D}\pi(x)f(x)=\mathbb{E}_{x\sim \pi} f(x)$
* This is a strong law of large numbers for Markov chains!

**Computing Expectations with MCMC**
* Joint sample at time $t$ depends on sample at time $t-1$
* Thus the law of large numbers (and sample complexity bounds such as Hoeffding's inequality) **do not apply**
* Use MCMC sampler to obtain samples
$\mathbf{X}^{(1)},...,\mathbf{X}^{(T)}$
* To let the Markov chain ''burn in'', ignore the first $t_0$ samples, and approximate
$\mathbb{E}[f(\mathbf{X})]\approx \frac{1}{T-t_0}\sum_{\tau=t_0+1}^T f(\mathbf{X}^{(\tau)})$
* Establishing convergence rates generally very difficult

**MCMC for Continuous RVs**
* MCMC techniques can be generalized to continuous random variables / vectors
* We focus on positive distributions w.l.o.g. written as
$p(\mathbf{x})=\frac{1}{Z}exp(-f(\mathbf{x}))$
where $f$ is called an energy function
* Distributions $p$ s.t. $f$ is convex are called **log-concave**
* Example: Bayesian logistic regression
$p(\theta|x_{1:n},y_{1:n})=\frac{1}{Z}p(\theta)p(y_{1:n}\theta,x_{1:n})=\frac{1}{Z}exp(-logp(\theta)-logp(y_{1:n}\theta,x_{1:n}))$
where $f(\theta)=\lambda\|\theta\|_2^2+\sum_{i=1}^nlog(1+exp(-y;\theta^Tx_i))+const.$

**Recall: Metropolis Hastings**
* 1) Proposal distribution $R(X’|X)\quad$ $Q(x)=exp(-f(x))$
    * Given $X_t=x$, sample “proposal” $x’\sim R(X’|X=x)$
* 2) Acceptance distribution:
    * Suppose $X_t=x$
    * With probability $\alpha=min\{1,\frac{R(x|x')}{R(x'|x)}exp(f(x)-f(x'))\}\quad$  set $X_{t+1}=x’$
    * With probability $1-\alpha$, set $X_{t+1}=x$
* What proposals $R$ should we use?

**Proposals**
* Open option: $R(x'|x)=\mathcal{N}(x';x;\tau I)$
$x'=x+\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,\tau I)$
* Acceptance probability?
    * *Note*: $\frac{\mathcal{N}(x|x',\tau I)}{\mathcal{N}(x'|x,\tau I)}=1$
    * so that $\alpha=min\{1,exp(f(x)-f(x'))\}$
    if $f(x')<f(x) \Leftrightarrow Q(x')>Q(x)\: \rightarrow \alpha=1$
    if $f(x')>f(x) \rightarrow 0<\alpha<1$
* Simple update, but “uninformed” direction

**Improved Proposals**
* Can take gradient information into account to prefer proposals into regions with higher density
$R(x'|x)=\mathcal{N}(x';x-\tau \nabla f(x);2\tau I)$
$x'=x-\tau \nabla f(x)+\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,2\tau I)$
* The resulting sampler is called **Metropolis adjusted Langevin Algorithm** (MALA; a.k.a. Langevin Monte Carlo, LMC)

**Guarantees for MALA**
* It is possible to show that for log-concave distributions (e.g., Bayesian log. Regression), MALA efficiently converges to the stationary distribution (mixing time is polynomial in the dimension)
$Q(x)=\frac{1}{Z}exp(-f(x))$ is log-concave **iff** $f$ convex
* In fact, locally the function is allowed to be non-convex

**Improving Efficiency?**
* Both the proposal and acceptance step in MALA/LMC require access to the full energy function $f$
* For large data sets, that can be **expensive**
* Key idea: 
    * Use stochastic gradient estimates
    * Use decaying step sizes and skip accept/reject step
* $\rightarrow$ **Stochastic Gradient Langevin Dynamics (SGLD)**

**Stochastic Gradient Langevin Dynamics**
* Consider sampling from the Bayesian posterior 
$\theta\sim \frac{1}{Z}exp(logp(\theta)+\sum_{i=1}^nlogp(y_i|x_i,\theta))$
* SGLD produces (approximate) samples as follows:
    * Initialize $\theta_0$
    * For $t=0,1,2,...$ do 
        * $\epsilon_t\sim\mathcal{N}(0,2\eta_t I)$
        * $\theta_{t+1}=\theta_k=\eta_t(\nabla logp(\theta_t)+\frac{n}{m}\sum_{j=1}^m \nabla logp(y_{i_j}|\theta_t,x_{i_j}))+\epsilon_t$

**Guarantees for SGLD**
* **SGLD = SGD + Gaussian noise**
* Can **guarantee convergence** to the stationary distribution (under some assumptions) as long as
$\eta_t\in \Theta(t^{-1/3})$
* In practice, one often uses **constant step sizes** to accelerate mixing (but needs tuning)
* Can improve performance via **preconditioning** (cf. Adagrad etc. for optimization)

**Outlook: Hamiltonian Monte Carlo (HMC)**
* Often, performance of (S)GD can be improved by adding a **momentum term**
* As SGLD/MALA can be seen as a sampling-based analogue of SGD, a similar analogue for (S)GD with momentum is the **Hamiltonian Monte Carlo algorithm**

**Summary: MCMC**
* **Markov Chain Monte Carlo** methods simulate a carefully designed Markov Chain to approximately sample from an intractable distribution
* Can be used for Bayesian learning
* For continuous distributions can make use of **(stochastic) gradient information** in the proposals
* Guaranteed, **efficient convergence for log-concave densities** (e.g., Bayesian logistic regression)
* In general, can guarantee convergence to the target distribution (in constrast to VI); however, for general distributions, convergence / mixing may be **slow**
* $\rightarrow$Tradeoff between accuracy and efficiency
