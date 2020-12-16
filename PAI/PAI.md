
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Probabilistic Artificial Intelligience](#probabilistic-artificial-intelligience)
  - [Introduction and probability](#introduction-and-probability)
    - [Lecture Notes](#lecture-notes)
  - [Bayesian Learning](#bayesian-learning)
    - [Lecture Notes](#lecture-notes-1)

<!-- /code_chunk_output -->
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