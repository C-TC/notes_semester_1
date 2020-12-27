## Introduction to Reinforcement Learning
### Lecture Notes
We will start with RL in **finite state/action spaces**, and later discuss how to scale to complex domains
**Learning to Act in Unknown Environments**
* Learn a mapping from (seq. of) actions to rewards
* **Credit assignment problem**: which actions got me to the large reward?

**Reinforcement learning**
* Agent actions change the state of the world (in contrast to supervised learning)
* Assumption: States change according to some (unknown) MDP!

**Recall: Markov Decision Processes**
* An (finite) MDP is specified by 
    * A set of **states** $X=\{1,...,n\}$
    * A set of **actions** $A=\{1,...,m\}$
    * **Transition probabilities**
    $P(x'|x,a)=Prob(Next \:state=x'|Action\: a \:in\: state\: x)$
    * A **reward function** $r(x, a)$
    Reward can be random with mean $r(x, a)$;
    Reward may depend on $x$ only or $(x, a, x’)$ as well.
* Here: Goal is to maximize $\sum_{t=0}^\infin \gamma^t r(x_t,a_t)$
* Observed state transitions and rewards let you learn the underlying MDP!

**Recall: Planning in MDPs**
* Deterministic Policy: $\pi: X\to A$
* Randomized Policy: $\pi: X\to P(A)$
* Incduces a Markov chainL $X_0,X_1,...,X_t,...$ with transition probabilities
$P(X_{t+1}=x'|X_t=x)=P(x'|x,\pi(x))$
For randomized policies: $P(X_{t+1}=x'|X_t=x)=\sum_a \pi(a|x)P(x'|x,a)$
* Expected value $J(\pi)=\mathbb{E}[r(X_0,\pi(X_0))+\gamma r(X_1,\pi(X_1))+\gamma^2 r(X_2,\pi(X_2))+...]$
where $\gamma\in [0,1)$ is discounted factor
* **value function**
$V^\pi(x)=J(\pi|X_0=x)=\mathbb{E}[\sum_{t=0}^ \infin \gamma^tr(X_t,\pi(X_t))|X_0=x]$

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

**Reinforcement Learning**
* RL is different from supervised learning
    * The data we get is not i.i.d.
    * In reinforcement learning, the data we get depends on our actions!
    * Some actions have higher rewards than others!
* Exploration—Exploitation Dilemma: Should we
    * **Explore**: gather more data to avoid missing out on a potentially large reward?
    * **Exploit**: stick with our current knowledge and build an optimal policy for the data we’ve seen?

**Two basic approaches to RL**
1) **Model-based RL**
    * Learn the MDP
        * Estimate transition probabilities $P(x’ | x, a)$
        * Estimate reward function $r(x, a)$
    * Optimize policy based on estimated MDP
2) **Model-free RL**
    * Estimate the value function directly;
    * Policy gradient methods;
    * Actor-critic methods.

**Off-policy vs on-policy RL**
* **On-policy RL**
    * Agent has full control over which actions to pick
    * Can choose how to trade exploration and exploitation
* **Off-policy RL**
    * Agent has no control over actions, only gets observational data (e.g., demonstrations, data collected by applying a different policy, …)

**Learning the MDP**
* Need to estimate
    * transition probabilities $P(X_{t+1}=x'|X_t=x,A=a)=\theta_{x'|x,a}$
    * Reward function $r(X=x,A=a)=r_{x,a}$
* E.g., using maximum likelihood estimation
* **Data set**: $\tau=(x_0,a_0,r_0,x_1,a_1,r_1,...,x_{T-1},a_{T-1},r_{T-1},x_T)$
Offen, multiple episodes $\tau^{(1)},\tau^{(2)},...,\tau^{(k)}$
$\rightarrow D=\{(x_0,a_0,r_0,x_1),(x_1,a_1,r_1,x_2),...\}$
* Estimate **transitions**:
$P(X_{t+1}|X_t,A)\approx \frac{Count(X_{t+1},X_t,A)}{Count(X_t,A)}$
where $Count(X_{t+1},X_t,A)=|\{i:(x_i=x,a_i=a,r_i,x_{i+1}=x')\in D\}|$
* Estimate **rewards**:
$r(x,a)\approx \frac{1}{N_{x,a}}\sum_{t:X_t=x,A_t=a}R_t$

**Exploration-Exploitation Dilemma**
* Always pick a random action?
    * Will eventually* correctly estimate all probabilities and rewards 
    * May do **extremely poorly** in terms of rewards! 
* Always pick the best action according to current knowledge?
    * Quickly get some reward
    * Can get stuck in **suboptimal action**! 
* Balance exploration and exploitation (more later)

**Trading Exploration and Exploitation**
* $\varepsilon_t$ greedy
    * With probability $\varepsilon_t$ : Pick random action
    * With probability $(1-\varepsilon_t)$: Pick best action
* If sequence $\varepsilon_t$ satisfies Robbins Monro (RM) conditions then will converge to optimal policy with probability 1
$\sum_t \varepsilon_t=\infin,\sum_t \varepsilon_t^2<\infin,e.g. \varepsilon_t=\frac{1}{t}$
* Simple, often performs fairly well
* **Doesn’t** quickly eliminate clearly suboptimal actions

**The Rmax Algorithm [Brafman & Tennenholz ‘02]**
* Optimism in the face of uncertainty!
**requires** $r(x,a)\leq R_{max}\: \forall x,a$
* If you don't know $r(x,a)$
    * Set it to $R_{max}$
* If you don't know $P(x'|x,a)$
* Set $P(x^*|x,a)=1$ where $x^*$ is a **''fairy tale''** state:
$P(x^*|x^*,a)=1,\forall a$
$r(x^*a)=R_{max},\forall a$

**Implicit Exploration Exploitation in Rmax**
* Never need to explicitly choose whether we’re exploring or exploiting!
* Can rule out clearly suboptimal actions very quickly

**The Rmax algorithm**
* Input: Starting state $x_0$, discount factor $\gamma$
* Initially:
    * Add fairy tale state $x^*$ to MDP
    * Set $r(x, a) = R_{max}$ for all states $x$ and actions $a$
    * Set $P(x^* | x, a) = 1$ for all states $x$ and actions $a$
    * Choose optimal policy for $r$ and $P$
* Repeat:
    * Execute policy $\pi$
    * For each visited state action pair $x, a$, update $r(x, a)$
    * Estimate transition probabilities $P(x'| x, a)$
    * If observed “enough” transitions / rewards, recompute policy $\pi$ according to current model $P$ and $r$

**How much is “enough”?**
How many samples do we need to accurately estimate **P(x'|x, a)** or **r(x,a)**?
* Hoeffding bound:
    * $Z_1,...,Z_n$ i.i.d. samples with mean $\mu$ and bounded in $[0,C]$
    $P(|\mu-\frac{1}{n}\sum_{i=1}^nZ_i|>\varepsilon)\leq 2exp(-2n\varepsilon^2/C^2)$
* e.g. $\hat{r}(x,a)=\frac{1}{n}\sum_{i=1}^nr_i, C=R_{max}$
$\Rightarrow P(|\hat{r}(x,a)-r(x,a)|>\varepsilon)\leq 2exp(-2n\varepsilon^2/R_{max}^2)$
if we want that $P(|\hat{r}(x,a)-r(x,a)|\leq\varepsilon)\geq 1-\delta$
it suffices that $2exp(-2n\varepsilon^2/R_{max}^2)\leq \delta \Rightarrow n\in O(\frac{R_{max}}{\varepsilon^2}log\frac{1}{\delta})$

**Exploration—Exploitation Lemma**
* Theorem: Every $T$ timesteps, w.h.p., $R_{max}$ either
    * Obtains near-optimal reward, or
    * Visits at least one unknown state-action pair
* $T$ is related to the mixing time of the Markov chain of the MDP induced by the optimal policy

**Performance of Rmax [Brafman & Tennenholz]**
* Theorem:
With probability $1-\delta$, $R_{max}$ will reach an $\varepsilon$-optimal policy in a number of steps that is polynomial in $|X|, |A|, T, 1/\varepsilon,log(1/\delta), R_{max}$

**Problems of model-based RL?**
* Memory required: For each $x,x'\in X$ and $a\in A$, need to store $\hat{P}(x'|x,a)$ and $\hat{r}(x,a)$
* Computation time: Need to solve est. MDP, e.g. using value/policy iteration. For $R_{max}$, have to do this possibly $n\cdot m $ times(i.e. when learned ''enough'' about $(x,a)$ pair)

**Warm-up: Value estimation**
* Given any policy $\pi$, want to estimate its value function $V^\pi(x)$
$V^\pi(x)=r(x,\pi(x))+\gamma \sum_{x'}P(x'|x,\pi(x))V^\pi(x')$
* Suppose we follow $\pi$ and obs. $(x,a,r,x')$
Further, assume we know $V^\pi(x)$
$V^\pi(x)=\mathbb{E}_{X',R}[R+\gamma V^\pi(X')|x,a]$
This suggests the following algorithm. Init. $V^\pi_0(x)$ somehow. At step $t$, obs. trans. $(x,a,r,x')$
update $V^\pi_{t+1}(x):=r+\gamma V_t^\pi(x')$
To reduce variance: instead $V^\pi_{t+1}(x)=(1-\alpha)V^\pi_t(x)+\alpha(r+\gamma V^\pi_t(x'))$

**Temporal Difference (TD)-Learning**
* Follow policy pi to obtain a transition $(x,a,r,x’)$
* Update value estimate using **bootstrapping**
$V(x)\leftarrow (1-\alpha_t)V(x)+\alpha_t(r+\gamma V(x'))$
* **Theorem**: If learning rate $\alpha_t$ satisfies
$\sum_t \alpha_t=\infin,\sum_t \alpha_t^2<\infin,e.g \: \alpha_t=\frac{1}{t}$
and all state-action pairs are chosen infinitely often, then $V$ converges to $V^\pi$ with probability 1
* How can we find the optimal policy?

**Model free RL**
* Recall:
    1. Optimal value function $V^*(x)\to$ policy $\pi^*$
    2. For optimal value function it holds:
    $V^*(x)=max_a Q^*(x,a)$
    where $Q^*(x,a)=r(x,a)+\gamma\sum_{x'}P(x'|x,a)V^*(x')$
* Key idea: Estimate $Q*(x, a)$ directly from samples!

**Q-learning**
* Estimate 
$Q^*(x,a)=r(x,a)+\gamma\sum_{x'}P(x'|x,a)V^*(x')$
$V^*(x)=max_aQ^*(x,a)$
* Surpose we
    * Have initial estimate of $Q(x, a)$
    * observe transition $x, a, x'$ with reward $r$
    $Q(x,a)\leftarrow(1-\alpha_t)Q(x,a)+\alpha_t(r+\gamma max_{a'}Q(x',a'))$
* **Theorem**: If learning rate $\alpha_t$ satisfies
$\sum_t \alpha_t=\infin,\sum_t \alpha_t^2<\infin,e.g \: \alpha_t=\frac{1}{t}$
and all state-action pairs are chosen infinitely often, then $Q$ converges to $Q^*$ with probability 1
* How can we trade off exploration and exploitation?

**Convergence of Optimistic Q-learning [Even-dar & Mansour ’02]**
* Similar to $R_{max}$:
Initialize $Q(x,a)=\frac{R_{max}}{1-\gamma}\prod_{t=1}^{T_{init}}(1-\alpha_t)^{-1}$
* **Theorem**: With prob. $1-\delta$, optimistic Q-learning obtains an e-optimal policy after a number of time steps that is polynomial in $|X|, |A|, 1/\varepsilon$ and $log(1/\delta)$
* At every step, greedily pick $a_t\in argmax_a Q(x_t,a)$

**Properties of Q-learning**
* Memory required: Leep track of $Q(x,a)\in \mathbb{R}^{n\times m}$
* Computation time: Per step: need to eval $V(x')=max_{a'}Q(x',a')$

**Key challenge: Scaling Up!**
* MDP and RL polynomial in $|A|$ and $|X|$. Problem in:
    * Structured domains (chess, multiagent planning, …): 
    $|X|$, $|A|$ exponential in #agents, state variables, …
    * Continuous domains ($|A|$ and $|X|$ infinite)
    * POMDPs (as belief-state MDPs)
* $\to$Learning / approximating value functions (regression)

## Reinforcement Learning via Function Approximation
### Lecture Notes
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

**Recall: value & action-value (Q) functions**
* Given fixed policy $\pi$, we have:
    * Value function:
    $V^\pi(x)=r(x,\pi(x))+\gamma\sum_{x'}P(x'|x,\pi(x))V^\pi(x')$
    * Action-value (Q) function:
    $Q^\pi(x,a)=r(x,a)+\gamma\sum_{x'}P(x'|x,a)V^\pi(x')=r(x,a)+\gamma\sum_{x'}P(x'|x,a)Q^\pi(x',\pi(x'))$
* For the optimal policy it holds:
$V^*(x)=max_a[r(x,a)+\gamma\sum_{x'}P(x'|x,a)V^*(x')]$
$Q^*(x,a)=r(x,a)+\gamma\sum_{x'}P(x'|x,a)V^\pi(x')=r(x,a)+\gamma\sum_{x'}P(x'|x,a)max_{a'}Q^*(x',a')$

**Off-policy vs on-policy RL**
* **On-policy RL**
    * Agent has full control over which actions to pick
    * Can choose how to trade exploration and exploitation
* **Off-policy RL**
    * Agent has no control over actions, only gets observational data (e.g., demonstrations, data collected by applying a different policy, …)

**Temporal Difference (TD)-Learning**
* Follow policy pi to obtain a transition $(x,a,r,x’)$, $a=\pi(x)$
* Update value estimate using **bootstrapping**
$\hat{V}^\pi(x)\leftarrow (1-\alpha_t)\hat{V}^\pi(x)+\alpha_t(r+\gamma \hat{V}^\pi(x'))$
* **Theorem**: If learning rate $\alpha_t$ satisfies
$\sum_t \alpha_t=\infin,\sum_t \alpha_t^2<\infin,e.g \: \alpha_t=\frac{1}{t}$
and all state-action pairs are chosen infinitely often, then $\hat{V}^\pi$ converges to $V^\pi$ with probability 1
* TD-Learning requires $a$ is picked by $\pi\to$  on-policy

**Off-policy Value Estimation**
* At state $x$ pick action $a$ to obtain a transition $(x,a,r,x')$
* Update value estimate using **bootstrapping**
$\hat{Q}^\pi(x)(x,a)\leftarrow(1-\alpha_t)\hat{Q}^\pi(x,a)+\alpha_t(r+\gamma \hat{Q}^\pi(x',\pi(x')))$
* **Theorem**: If learning rate $\alpha_t$ satisfies
$\sum_t \alpha_t=\infin,\sum_t \alpha_t^2<\infin,e.g \: \alpha_t=\frac{1}{t}$
and all state-action pairs are chosen infinitely often, then $\hat{Q}^\pi$ converges to $Q^\pi$ with probability 1
* Action $a$ need not be picked via $\pi\to$ off-policy possible

**RL via Q-learning**
* $\hat{Q}^*(x)(x,a)\leftarrow(1-\alpha_t)\hat{Q}^*(x,a)+\alpha_t(r+\gamma \hat{Q}^*(x',a'))$
* **Theorem**: If learning rate $\alpha_t$ satisfies
$\sum_t \alpha_t=\infin,\sum_t \alpha_t^2<\infin,e.g \: \alpha_t=\frac{1}{t}$
and all state-action pairs are chosen infinitely often, then $\hat{Q}^*$ converges to $Q^*$ with probability 1
* Action $a$ need not be picked via $\pi\to$ off-policy possible

**Key challenge: Scaling Up!**
* MDP and RL polynomial in $|A|$ and $|X|$. Problem in:
    * Structured domains (chess, multiagent planning, …): 
    $|X|$, $|A|$ exponential in #agents, state variables, …
    * Continuous domains ($|A|$ and $|X|$ infinite)
    * POMDPs (as belief-state MDPs)
* $\to$Learning / approximating value functions (regression)

**TD-Learning as SGD**
* $V^\pi(x)\leftarrow (1-\alpha_t)\hat{V}^\pi(x)+\alpha_t(r+\gamma \hat{V}^\pi(x'))$
$\bar{l}_2(V;x,r):=\frac{1}{2}(V-r-\gamma\mathbb{E}_{x'|x,\pi(x)}\hat{V}^\pi(x'))^2$
$\nabla_V \bar{l}_2(v;x,r)=V-r-\gamma\mathbb{E}_{x'|x,\pi(x)}\hat{V}^\pi(x')$
obs. $x'\sim P(x'|x,\pi(x))$
$\Rightarrow V-r-\gamma\hat{V}^\pi(x'):=\delta$ TD-error, is unbiased estimate of $\nabla_V \bar{l}_2(v;x,r)$
SGD: $V\leftarrow V-\alpha_t\delta$
$\hat{V}^\pi(x)\leftarrow=\hat{V}^\pi(x)-\alpha_t(\hat{V}^\pi(x)-r-\gamma\hat{V}^\pi(x'))=(1-\alpha_t)\hat{V}^\pi(x)+\alpha_t(r+\gamma \hat{V}^\pi(x'))$

**Can view TD-learning as SGD!**
* Tabular TD-learning update rule can be viewed as an instance of **stochastic (semi-)gradient descent on the squared loss**
* $l_2(\theta;x,x',r)=\frac{1}{2}(V(x;\theta)-r-\gamma V(x';\theta_{old}))^2$
* $r+\gamma V(x';\theta_{old})$ is y label(a.k.a target)
    * Parameters are entries in value vector
    * Experience / transition data sampled on-policy
* Bootstrapping means to use “old” value estimates as labels (a.k.a. targets)
* Same insight applies to learning the (optimal) action-value function
* $\to$ path towards parametric function approximation!

**Parametric value function approximation**
*  To scale to large state spaces, learn an **approximation** of (action) value function $V(x;\theta)\: or\:Q(x,a;\theta) $
*  Examples:
    *  Linear function approximation $Q(x,a;\theta)=\theta^T\phi(x,a)$
    where $\phi(x,a)$ are a set of (hand-designed) features
    *  **(Deep) Neural networks $\to$ Deep RL**

**Recall: Deep Learning**
* Fitting nested nonlinear functions (neural nets)
$f(\mathbf{x};\mathbf{w})=\varphi_l(\mathbf{W}_l\varphi_{l-1}(\mathbf{W}_{l-1}(...\varphi_1(\mathbf{W}_1\mathbf{x}))))$
* to data by (approximately) solving
$\mathbf{w}^*=argmin_{\mathbf{w}}\sum_{i=1}^N l(y_i,f(\mathbf{x}_i;\mathbf{w}))$
via **stochastic gradient descent**.
Can obtain gradient via chain-rule (**backpropagation**)

**Gradients for Q-learning with function approximation**
* Example: linear function approximation
$\hat{Q}(x,a;\theta)=\theta^T\phi(x,a)$
* After observing transition $(x,a,r,x')$, update via gradient of
$l_2(\theta;x,a,r,x')=\frac{1}{2}(Q(x,a;\theta)-r-\gamma max_{a'}Q(x',a';\theta_{old}))^2=\frac{1}{2}\delta^2$
$\begin{aligned}
\theta&\leftarrow\theta-\alpha_t\nabla l_2(\theta;x,a,r,x')\\
&=\theta-\alpha_t\delta\cdot\nabla_\theta Q(x,a;\theta)\\
&=\theta-\alpha_t\delta\phi(x,a)\\
\end{aligned}$

**Q-learning with function approximation**
* Straight forward generalization of tabular Q learning to function approximation suggests online algorithm:
* Until converged
    * In state $x$, pick action $a$
    * Observe $x'$, reward $r$
    * Update $\theta\leftarrow \theta-\alpha_t\delta\nabla_\theta Q(x,a;\theta)$
    where $\delta:=Q(x,a;\theta)-r-\gamma max_{a'}Q(x',a';\theta)$
* This basic algorithm is typically rather **slow**

**Neural Fitted Q-iteration / DQN [Riedmiller ‘05, Mnih et al ’15]**
* To accelerate Q-learning with (neural net) function approximation:
    * use ''experience replay''
        * Maintain data set D of observed transitions $(x,a,x', r)$
    * clone network to maintain constant ''target'' values across episodes
    $L(\theta)=\sum_{(x,a,r,x')\in D}(r+\gamma max_{a'}Q(x',a';\theta^{old})-Q(x,a;\theta))^2$

**Increasing stability: Double DQN [van Hasselt et al. 2015]**
* **Standard DQN**:
$L(\theta)=\sum_{(x,a,r,x')\in D}(r+\gamma max_{a'}Q(x',a';\theta^{old})-Q(x,a;\theta))^2$
* Suffers from ''maximization bias''
* **Double DQN**: current network for evaluating the argmax
$L^{DDQN}(\theta)=\sum_{(x,a,r,x')\in D}(r+\gamma max_{a'}Q(x',a^*(\theta);\theta^{old})-Q(x,a;\theta))^2$
where $a^*(\theta):=argmax_{a'}Q(x',a';\theta)$

**Convolutional neural networks**
* Convolutional neural networks are ANNs for **specialized applications** (e.g., image recognition) 
* The hidden layer(s) closest to the input layer **shares parameters**: Each hidden unit only depends on all ''closeby'' inputs (e.g., pixels), and weights constrained
to be identical across all units on the layer 
* This reduces the number of parameters, and encourages robustness against  small amounts of) translation
* The weights can still be optimized via backpropagation

**Dealing with large action sets**
* Q-learning implicitly defines a policy via
$a_t=argmax_a Q(x_t,a;\theta)$
* For large / continuous action spaces, this is **intractable**

**Policy search methods**
* Learning a **parameterized** policy
$\pi(x)=\pi(x;\theta)$
* For episodic tasks (i.e., can reset ''agent'') can compute expected reward by ''rollouts'' (Monte Carlo forward sampling; $\to$ ''on policy'')
$\tau^{(0)},...,\tau^{(m)}\sim \pi_\theta\quad$; $\tau^{(i)}=(x_0^{(i)},a_0^{(i)},v_0^{(i)},x_1^{(i)},...,x_T^{(i)})$
$r(\tau^{(i)})=\sum_{t=1}^T\gamma^tr_t^{(i)}\to J(\theta)\approx \frac{1}{m}\sum_{i=1}^mr(\tau^{(i)})$
* $\to$ Find optimal parameters through global optimization
$\theta^*=argmax_\theta J(\theta)$

**Policy gradients**
* Objective: maximize
$J(\theta)=\mathbb{E}_{x_{0:T},a_{0:T}\sim \pi_\theta}\sum_{t=0}^T \gamma^tr(x_t,a_t)=\mathbb{E}_{\tau\sim\pi_\theta}r(\tau)$
* How can we obtain gradients w.r.t. $\theta$?

**Obtaining policy gradient**
* Theorem: It holds* that
$\nabla J(\theta)=\nabla \mathbb{E}_{\tau\sim\pi_\theta}r(\tau)=\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\nabla log \pi_\theta(\tau)]$
* Proof:
$\begin{aligned}
\nabla J(\theta)&=\nabla\int \pi_\theta(\tau)r(\tau)d\tau\\
&=\int\nabla\pi_\theta(\tau)r(\tau)d\tau\\
&=\int r(\tau)\pi_\theta(\tau)\nabla log\pi_\theta(\tau)d\tau\\
&=\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\nabla log\pi_\theta(\tau)]
\end{aligned}$
* *Note*: $\nabla log\pi_\theta(\tau)=\frac{\nabla \pi_\theta(\tau)}{\pi_\theta(\tau)}\Rightarrow \nabla\pi_\theta(\tau)=\pi_\theta(\tau)\nabla log \pi_\theta(\tau)$

**Exploiting the MDP structure**
* To obtain gradients for $J(\theta)$, need to compute
$\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\nabla log \pi_\theta(\tau)]$
* From the MDP, we have $r(\tau)=\sum_{t=0}^T\gamma^tr(x_t,a_t)$
$\pi_\theta(\tau)=P(x_0)\prod_{t=0}^T\pi(a_t|x_t;\theta)P(x_{t+1}|x_t,a_t)$
* Thus
$\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\nabla log \pi_\theta(\tau)]=\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\sum_{t=0}^T\nabla log\pi(a_t|x_t;\theta)]$

**Reducing variance**
* Even though the gradients obtained via
$\nabla J(\theta)=\nabla\mathbb{E}_{\tau\sim\pi_\theta}r(\tau)=\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\nabla log \pi_\theta(\tau)]$
are **unbiased**, they typically exhibit **very large variance**
* Can reduce the variance using so-called **baselines**.
* Key insight: it holds that
$\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\nabla log \pi_\theta(\tau)]=\mathbb{E}_{\tau\sim\pi_\theta}[(r(\tau)-b)\nabla log \pi_\theta(\tau)]$

**Proof**
* $\mathbb{E}_{\tau\sim\pi_\theta}[(r(\tau)-b)\nabla log \pi_\theta(\tau)]=\mathbb{E}_{\tau\sim\pi_\theta}[r(\tau)\nabla log \pi_\theta(\tau)]-\mathbb{E}_{\tau\sim\pi_\theta}[b\nabla log \pi_\theta(\tau)]$
* $\mathbb{E}_{\tau\sim\pi_\theta}[b\nabla log \pi_\theta(\tau)]=b\int\pi_\theta(\tau)\nabla_\theta log\pi_\theta(\tau)d\tau=b\int\nabla_\theta\pi_\theta(\tau)d\tau=b\nabla_\theta\int\pi_\theta(\tau)d\tau=0$
* *Note*: $\nabla\pi_\theta(\tau)=\pi_\theta(\tau)\nabla log \pi_\theta(\tau)$

**State-dependent baselines**
* Similarly, one can show that
$\mathbb{E}_{\tau\sim\pi_\theta}[\sum_{t=0}^Tr(\tau)\nabla log\pi(a_t|x_t;\theta)]=\mathbb{E}_{\tau\sim\pi_\theta}[\sum_{t=0}^T(r(\tau)-b(\tau_{0:t-1}))\nabla log\pi(a_t|x_t;\theta)]$
* For example, can choose $b(\tau_{0:t-1})=\sum_{t'=0}^{t-1}\gamma^{t'}r_{t'}$
and thus $\nabla J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[\sum_{t=0}^T\gamma^tG_t\nabla log\pi(a_t|x_t;\theta)]$
where $G_t=\sum_{t'=t}^T\gamma^{t'-t}r_{t'}$ is the reward to go following action $a_t$

**REINFORCE [Williams’92]**
* Input: $\pi(a|x;\theta)$
    1. Initialize policy weights $\theta$
    2. Repeat:
        1. Generate an episode (rollout): $X_0,A_0,R_0,X_1,A_1,R_1,...,X_T,A_T,R_T$
        2. For $t = 0, ...T$:
            Set $G_t$ to the return from step $t$
            Update $\theta=\theta+\eta\gamma^tG_t\nabla_\theta log\pi(A_t|X_t;\theta)$

**Further variance reduction**
* Basic REINFORCE gradient estimate:
$\nabla J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[\sum_{t=0}^T\gamma^tG_t\nabla log\pi(a_t|x_t;\theta)]$
* Can further reduce variance via stronger baselines
$\nabla J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[\sum_{t=0}^T\gamma^t(G_t-b_t(x_t))\nabla log\pi(a_t|x_t;\theta)]$
* Example: Mean over returns
$b_t(x_t):=b_t=\frac{1}{T}\sum_{t=0}^{T-1}G_t$

## Deep RL with policy gradients and actor-critic methods
### Lecture Notes
**Value Functions and Policies**
**Recall: value & action-value (Q) functions**
**New concept: Advantage function**
* For a given policy $\pi$, can consider the advantage of
playing action $a$ in state $x$:
$A^\pi(x,a)=Q^\pi(x,a)-V^\pi(x)=Q^\pi(x,a)-\mathbb{E}_{a'\sim\pi(x)}Q^\pi(x,a')$
* $\forall \pi,x$: $max_aA^\pi(x,a)\geq 0$
$\pi^*$ is optimal$\Leftrightarrow \forall x,a,\:A^*(x,a)\leq 0$
Greedy policy w.r.t. $\pi$: $\pi_G(x)=argmax_a Q^\pi(x,a)=argmax_a A^\pi(x,a)$

**Temporal Difference (TD)-Learning**
**Off-policy Value Estimation**
**Model-free RL via Q-learning**
**Key challenge: Scaling Up!**
**Parametric value function approximation**
*  To scale to large state spaces, learn an **approximation** of (action) value function $V(x;\theta)\: or\:Q(x,a;\theta) $
*  Examples:
    *  Linear function approximation $Q(x,a;\theta)=\theta^T\phi(x,a)$
    where $\phi(x,a)$ are a set of (hand-designed) features
    *  **(Deep) Neural networks $\to$ Deep RL**
* Can update parameters by **minimizing squared loss on predicted ''bootstrapped'' targets** via SGD

**Neural Fitted Q-iteration / DQN [Riedmiller ‘05, Mnih et al ’15]**
**Dealing with large action sets**
**Policy search methods**
**Exploiting the MDP structure**
**REINFORCE [Williams’92]**

**Further improvements to policy gradients**
* Basic policy gradient methods are slow
* Improvements:
    * Natural gradients
    * Using value function estimates $\to$ actor critic methods
    * Regularization & constrained optimization
    * Off-policy variants
* Today:
    * Introduce basic actor critic algorithm
    * Review basic ideas behind an array of modern policy gradient methods (A2C/A3C, TRPO, PPO, DDPG, TD3, SAC)

**Reinterpreting score gradients**
* $\nabla J_T(\theta)=\mathbb{E}_{\tau\sim\pi(\theta)}[\sum_{t=0}^T\gamma^tG_t\nabla log\pi(a_t|x_t;\theta)$
$J(\theta)=\mathbb{E}_{\tau\sim\pi(\theta)}[\sum_{t=0}^\infin \gamma^tr_t]$;
$\begin{aligned}
\nabla J(\theta)&=lim_{T\to\infin}\nabla J_T(\theta)\\
&=\sum_{t=0}^\infin\mathbb{E}_\tau[\gamma^tG_t\nabla log\pi(a_t|x_t;\theta)]\\
&=\sum_{t=0}^\infin\mathbb{E}_{\tau_{t:\infin}}[\gamma^tG_t\nabla log\pi(a_t|x_t;\theta)]\\
&=\sum_{t=0}^\infin\mathbb{E}_{x_t,a_t}[\gamma^t\nabla log\pi(a_t|x_t;\theta)\mathbb{E}[G_t|x_t,a_t]]\\
&=\mathbb{E}_{\tau\sim\pi(\theta)}[\sum_{t=0}^\infin\gamma^tQ(x_t,a_t)\nabla log\pi(a_t|x_t;\theta)]
\end{aligned}$
* *Note*: $\tau_{t:\infin}=(x_t,a_t,r_t,x_{t+1},...)$

**Actor Critic methods**
* Can use value function estimates in conjunction with policy gradient methods:
$\begin{aligned}
\nabla J(\theta)&=\mathbb{E}_{\tau\sim\pi(\theta)}[\sum_{t=0}^\infin\gamma^tQ(x_t,a_t;\theta_Q)\nabla log\pi(a_t|x_t;\theta)]\\
&=\int \rho^\theta(x)\mathbb{E}_{a\sim\pi_\theta(x)}[Q(x,a;\theta_Q)\nabla log\pi(a|x;\theta)]dx\\
&=\mathbb{E}_{x\sim\rho^\theta,a\sim\pi_\theta(x)}[Q(x,a;\theta_Q)\nabla log\pi(a|x;\theta)]\\
&=:\mathbb{E}_{(x,a)\sim\pi_\theta}[Q(x,a;\theta_Q)\nabla log\pi(a|x;\theta)]
\end{aligned}$
* *Note*: $\rho^\theta(x)=\sum_{t=0}^\infin \gamma^tp_\theta(x_t=x)$ is the discounted state occupancy measure

**Actor Critic methods**
* Can use value function estimates in conjunction with policy gradient methods [a.k.a. **policy gradient thm.**]:
$\nabla J(\theta_\pi)=\mathbb{E}_{(x,a)\sim\pi_\theta}[Q(x,a;\theta_Q)\nabla log\pi(a|x;\theta)]$
* Allows application in the **online (non-episodic)** setting
* At time t, upon observing a transition $(x,a,r,x')$, update:
$\theta_\pi\leftarrow \theta_\pi+\eta_t Q(x,a;\theta_Q)\nabla log\pi(a|x;\theta)$
$\theta_Q\leftarrow \theta_Q-\eta_t(Q(x,a;\theta_Q)-r-\gamma Q(x',\pi(x';\theta_\pi);\theta_Q))\nabla Q(x,a;\theta_Q)$
* Under “**compatibility conditions**” guaranteed to improve

**Outlook: Variance reduction via baselines**
* Can improve convergence performance via variance reducing baselines (as in REINFORCE)
$\theta_\pi\leftarrow \theta_\pi+\eta_t[Q(x,a;\theta_Q)-V(x;\theta_V)]\nabla log\pi(a|x;\theta)$
where $Q(x,a;\theta_Q)-V(x;\theta_V)$ is Advantage function estimate $\to$ A2C algorithm
* This technique can be combined with Monte-Carlo Return estimation (blending between REINFORCE and actor critic methods $\to$ GAAC algorithm)

**Outlook: Efficient implementations**
* Actor critic methods can be efficiently implemented in paralled $\to$ E.g., Asynchronous Advantage Actor Critic (A3C, Mnih et al)

**Outlook: TRPO & PPO**
* Modern variants of policy gradient / actor critic methods
* Trust-region policy optimization (TRPO) [Schulman et al ‘17]
    * Sequentially optimizes a sequence of surrogate problems
    $\theta_{k+1}=argmax_\theta \hat{J}(\theta_k,\theta) \:s.t.\: KL(\theta\|\theta_k)\leq \delta$
    $\hat{J}(\theta_k,\theta)=\mathbb{E}_{x,a\sim\pi_{\theta_k}}[\frac{\pi(a|x;\theta)}{\pi(a|x;\theta)_k}A^{\pi_{\theta_k}}(x,a)]$
    * Guarantees monotonic improvement in $J(\theta)$
* Proximal Policy Optimization (PPO) [Schulman et al ’17]
    * Heuristic variant of TRPO (uses a certain clipped surrogate)
    * effective and widely used in practice

**Towards off-policy actor critic**
* All algorithms discussed so far are on-policy methods
* This often causes **sample inefficiency**
* Is it possible to train policy gradient methods in an **off-policy** fashion?

**Another approach to policy gradients**
* Our initial motivation was intractability of $max_{a'}Q(x',a';\theta^{old})$
in $L(\theta)=\sum_{(x,a,r,x')\in D}(r+\gamma max_{a'}Q(x',a';\theta^{old})-Q(x,a;\theta))^2$
* What if we *replace the exact maximum** by a **parametrized policy**?
$L(\theta)=\sum_{(x,a,r,x')\in D}(r+\gamma Q(x',\pi(x';\theta_\pi);\theta_Q^{old})-Q(x,a;\theta))^2$
* But how do we update our policy parameters $\theta_\pi$?

**Updating policy parameters**
* We want to follow the greedy policy
$\pi_G(x)=argmax_aQ(x,a;\theta_Q)$
* If we allow ''rich enough'' policies, this is equivalent* to
$\theta_\pi^*\in argmax_\theta\mathbb{E}_{x\sim\mu}[Q(x,\pi(x;\theta);\theta_Q)]$
where $\mu(x)>0$ “explores all states”
* Key idea: If we use differentiable approximation $Q(\cdot;\theta_Q)$ and differentiable deterministic policy $\pi(\cdot;\theta_\pi)$ can use chain rule (backpropagation) to obtain stochastic gradients!

**Computing gradients**
* Objective: $\theta_\pi^*\in argmax_\theta\mathbb{E}_{x\sim\mu}[Q(x,\pi(x;\theta);\theta_Q)]=argmax_\theta J(\theta)$
$\nabla J(\theta)=\mathbb{E}_{x\sim\mu}[\nabla_\theta Q(x,\pi(x;\theta);\theta_Q)]$
$\Rightarrow $can compute unbiased gradient estimate by sampling $x\sim\mu$
* From the chain rule
$\nabla_{\theta_\pi}Q(x,\pi(x;\theta_\pi);\theta_Q)=\nabla_a Q(x,a)|_{a=\pi(x;\theta_\pi)}\nabla_{\theta_\pi}\pi(x;\theta_\pi)$

**Exploration**
* Policy gradient methods rely on **randomized policies** for exploration
* The method we just discussed uses deterministic policies. How do we ensure **sufficient exploration**?
* Since method is off-policy, can **inject additional action noise** (e.g., Gaussian) to encourage exploration (akin to epsilon—greedy exploration)

**Deep Deterministic Policy Gradients (DDPG)**
* Init. $\theta_Q,\theta_\pi$, replay buffer $D=\{\}$; $\theta^{old}_Q=\theta_Q$ ; $\theta^{old}_\pi=\theta_\pi$
* Repeat
    * Observe state $x$; carry out action $a=\pi(x;\theta_\pi)+\varepsilon$
    * Execute action $a$; observe reward $r$ and next state $x'$
    * Store $(x,a,r,x')$ in $D$
    * If time to update
        * For some iterations do
            * Sample mini-batch B of transitions $(x,a,r,x')$ from $D$
            * For each, compute target $y=r+\gamma Q(x',\pi(x',\theta_\pi^{old}),\theta_Q^{old})$
            * $\theta_Q\leftarrow \theta_Q-\eta\nabla\frac{1}{|B|}\sum_{(x,a,r,x',y)\in B}(Q(x,a;\theta_Q)-y)^2$
            * $\theta_\pi\leftarrow \theta_\pi+\eta\nabla\frac{1}{|B|}\sum_{(x,a,r,x',y)\in B}Q(x,\pi(x;\theta_\pi);\theta_Q)$
            * $\theta_Q^{old}\leftarrow (1-\rho)\theta_Q^{old}+\rho \theta_Q;\:\theta_\pi^{old}\leftarrow (1-\rho)\theta_\pi^{old}+\rho \theta_\pi $

**Outlook: Twin Delayed DDPG (TD3)**
* Extends DDPG by using two critic networks, and evaluating the advantage with the smaller one ($\to$ to address **maximization bias** akin to Double-DQN)
* Applies delayed updates to actor network, which increases **stability**

**Dealing with randomized policies**
* In DDPG, had to inject random noise to ensure exploration 
Can we **directly allow randomized policies**?
* How about the **critic update**
$\theta_Q\leftarrow \theta_Q-\eta\nabla\frac{1}{|B|}\sum_{(x,a,r,x',y)\in B}(Q(x,a;\theta_Q)-y)^2$
where $y=r+\gamma Q(x',\pi(x',\theta_\pi^{old}),\theta_Q^{old})$
* For randomized policies: $(Q(x,a;\theta_Q)-y)^2=\mathbb{E}_{a'\sim\pi}(Q(x,a;\theta_Q)-y(a'))^2$
where we can obtain unbiased gradient estimates by sampling from $a'\sim\pi(x';\theta_\pi^{old})$
$\begin{aligned}
\nabla_{\theta_Q}\mathbb{E}_{a'\sim\pi}(Q(x,a;\theta_Q)-y(a'))^2&=\mathbb{E}_{a'}\nabla_{\theta_Q}(Q(x,a;\theta_Q)-y(a'))^2\\
&:=\mathbb{E}_{a'}\nabla_{\theta_Q}\delta^2(a')\\
&=2\delta(a')\nabla_{\theta_Q}Q(x,a;\theta_Q)
\end{aligned}$
* How about the **policy update** step?

**Reparametrization gradients**
* For deterministic policies, recall:
$\nabla_{\theta_\pi}Q(x,\pi(x;\theta_\pi);\theta_Q)=\nabla_a Q(x,a)|_{a=\pi(x;\theta_\pi)}\nabla_{\theta_\pi}\pi(x;\theta_\pi)$
* Suppose policy is **reparametrizable**, i.e., $a\sim\pi(x;\theta_\pi)$ is such that the action is generated by $a=\psi(x;\theta_\pi,\epsilon)$, where $\epsilon$ is an independent random variable
* Example: Gaussian policies $a=C(x;\theta_\pi)\epsilon+\mu(x;\theta_\pi)$
where $\epsilon\sim\mathcal{N}(0,I)$ [see variational inference lecture]
* Then $\nabla_{\theta_\pi}\mathbb{E}_{a\sim\pi_{\theta_\pi}}Q(x,a;\theta_Q)=\mathbb{E}_\epsilon\nabla_{\theta_\pi}Q(x,\psi(x;\theta_\pi,\epsilon);\theta_Q)=\mathbb{E}_\epsilon[\nabla_aQ(x,a;\theta_Q)|_{a=\psi(x;\theta_\pi,\epsilon)}\nabla_{\theta_\pi}\psi(x;\theta_\pi,\epsilon)]$
* Thus can obtain gradients for reparametrizable stochastic policies (applies beyond Gaussians)! 
**Outlook: Entropy regularization**
* One natural way to encourage exploration is to consider entropy regularized MDPs:
$J_\lambda(\theta)=J(\theta)+\lambda H(\pi_\theta)=\mathbb{E}_{(x,a)\sim\pi_\theta}[r(x,a)+\lambda H(\pi(\cdot|x))]$
* Thus, use entropy of action distribution to encourage exploration
* Can suitably define regularized (action)-value functions (called “soft” value functions)
* Can use reparametrization gradients to obtain the **Soft Actor Critic (SAC)**algorithm

**Overview: Policy gradient algorithms**
* **On-policy** policy gradient methods
    * REINFORCE: optimizes score-gradient using Monte-Carlo returns; high variance $\to$ need baselines
    * Actor Critic methods: use value function / advantage function estimate ($\to$ A2C, A3C); implement approximate (generalized) policy iteration
    * TRPO iteratively optimizes a surrogate objective within trust region; PPO is an effective heuristic variant
* **Off-policy** policy gradient methods
    * Importance weighted variants (not discussed here)
    * DDPG: combines DQN with reparametrization policy gradients
    * TD3: extension of DDPG to avoid maximization bias
    * SAC: variant of DDPG/TD3 for entropy regularized MDPs

## Model-based Deep RL
### Lecture Notes
**Recall: Deterministic Policy Gradients**
**Reparametrization gradients**
**Model-based Deep RL**
* So far, we have focused on model-free methods
* If we have an accurate model of the environment, we can use it for **planning**
* Learning a model can help dramatically **reduce the sample complexity** compared to model-free techniques

**Overview**
* We first provide the high-level ideas for **planning** according to a **known dynamics** model and reward
* We then discuss how to **learn** a dynamics model 
* Lastly, we discuss **exploration—exploitation** tradeoffs in the model-based setting

**Planning**
* There is a large literature on planning
    * **discrete and continuous** action spaces
    * **fully and partially observed** state spaces
    * with or without **constraints**
    * linear and non-linear transition models
    * ...
* Here we focus on planning in **continuous, fully observed** state spaces with **non-linear** transitions, **without constraints**

**Planning with a known deterministic model**
* To start, assume we have a **known deterministic** model for the reward and dynamics
$x_{t+1}=f(x_t,a_t)$
* Then, our objective becomes
$max_{a_{0:\infin}}\sum_{t=0}^\infin \gamma^tr(x_t,a_t)\quad s.t. \: x_{t+1}=f(x_t,a_t)$
* Cannot explicitly optimize over an infinite horizon

**Receding-horizon / Model-predictive control**
* Key idea: Plan over a **finite horizon $H$**, carry out first action, then **replan**
    * At each iteration $t$, observe $x_t$, 
    * Optimize performance over horizon $H$
    $max_{a_{t:t+H-1}}\sum_{\tau=t:t+H-1} \gamma^{\tau-t}r_\tau(x_\tau,a_\tau)\quad s.t. \: x_{\tau+1}=f(x_\tau,a_\tau)$
    * Carry out action $a_t$, then replan

**Solving the optimization problem**
* At each iteration, need to solve
$max_{a_{t:t+H-1}}\sum_{\tau=t:t+H-1} \gamma^{\tau-t}r_\tau(x_\tau,a_\tau)\quad s.t. \: x_{\tau+1}=f(x_\tau,a_\tau)$
* For deterministic models $f$, $x_\tau$ is determined by $a_{t:\tau-1}$
$x_{t+1}=f(x_t,a_t)$
$x_{t+2}=f(x_{t+1},a_{t+1})=f(f(x_t,a_t),a_{t+1})$
$\vdots$
$x_\tau=f(f(...f(x_t,a_t),a_{t+1})...,a_{\tau-1})=:x_\tau(a_{t:\tau-1})$
* Thus, at step $t$, need to maximize
$J_H(a_{t:t+H-1}):=\sum_{\tau=t:t+H-1} \gamma^{\tau-t}r_\tau(x_\tau(a_{t:\tau-1}),a_\tau)$

**How to optimize?**
* Need to optimize
$J_H(a_{t:t+H-1}):=\sum_{\tau=t:t+H-1} \gamma^{\tau-t}r_\tau(x_\tau(a_{t:\tau-1}),a_\tau)$
* For continuous actions, differentiable rewards and differentiable dynamics, can **analytically compute gradients** ($\to$ backpropagation through time)
* Challenges (especially for large $H$):
    * **Local minima**
    * **Vanishing / exploding gradients**
* $\to$ Often use heuristic global optimization methods

**Outlook: Random shooting methods**
* Sampling approach towards global optimization of
$J_H(a_{t:t+H-1}):=\sum_{\tau=t:t+H-1} \gamma^{\tau-t}r_\tau(x_\tau(a_{t:\tau-1}),a_\tau)$
* Generate m sets of **random samples** $a^{(i)}_{t:t+H-1}$
    * E.g., from a Gaussian distribution, cross-entropy method,...
* Pick the sequence $a^{(i^*)}_{t:t+H-1}$ that optimizes
$i^*=argmax_{i\in\{1,...,m\}}J_H(a_{t:t+H-1}^{(i)})$
* Side note: Monte-Carlo Tree Search used in AlphaZero can be seen as advanced variant of a shooting method

**Limitations of finite-horizon planning**
**Using a value estimate**
* Suppose we have access to (an estimate of) the value function $V$. Then we can consider
$J_H(a_{t:t+H-1}):=\sum_{\tau=t:t+H-1} \gamma^{\tau-t}r_\tau(x_\tau(a_{t:\tau-1}),a_\tau)+\gamma^HV(x_{t+H})$
* For $H=1$,
$a_t=argmax_a J_H(a)$ is simply the **greedy policy** w.r.t. $V$
* Can also optimize using gradient-based or global optimization (shooting) methods
* Can obtain value estimates using off-policy estimation (as discussed earlier)

**MPC for stochastic transition models?**
* At each iteration $t$, observe $x_t$,
* Optimize expected performance over horizon $H$
$max_{a_{t:t+H-1}}\mathbb{E}_{x_{t+1:t+H}}[\sum_{\tau=t:t+H-1}\gamma^{\tau-t}r_\tau+\gamma^HV(x_{t+H})|a_{t:t+H-1}]$
* Carry out action $a_t$, then replan

**Optimizing expected performance**
* For probabilistic transition models via MPC, need to optimize
$J_H(a_{t:t+H-1}):=\mathbb{E}_{x_{t+1:t+H}}[\sum_{\tau=t:t+H-1}\gamma^{\tau-t}r_\tau+\gamma^HV(x_{t+H})|a_{t:t+H-1}]$
* Computing this expectation exactly requires solving a **high-dimensional integral**
* One common approach:
Monte-Carlo **trajectory sampling**
* Suppose the transition model is **reparametrizable**, i.e., $x_{t+1}=f(x_t,a_t,\epsilon_t)$, where $\epsilon_t$ is **independent** of $a,x$
    * E.g., nonlinear dynamics with Gaussian noise
*In this case, $x_\tau$ is determined by $a_{t:\tau-1}$ and $\epsilon_{t:\tau-1}$ via
$x_\tau:=x_\tau(a_{t:\tau-1},\epsilon_{t:\tau-1})$
$:=f(f(...f(x_t,a_t,\epsilon_t),a_{t+1},\epsilon_{t+1})...,a_{\tau-1},\epsilon_{\tau-1})$
* $\to$ can obtain **unbiased estimates** of $J_H(a_{t:t+H-1})$ by 
$\hat{J}_H(a_{t:t+H-1})=\frac{1}{m}\sum_{i=1:m}\sum_{\tau=t:t+H-1}\gamma^{\tau-t}r_\tau(x_\tau(a_{t:\tau-1},\epsilon^{(i)}_{t:\tau-1}),a_\tau)+\gamma^HV(x_{t+H})$
* Optimize, e.g., via analytic gradients, or shooting methods

**Using parametrized policies**
* Instead of explicitly optimizing over $a_t,...,a_{t+H-1}$, can also optimize over **parametrized policies** (stochastic policies possible too via reparametrization)
$a_t=\pi(x_t,\theta)$
* The objective becomes
$J(\theta)=\mathbb{E}_{x_0\sim\mu}[\sum_{\tau=0:H-1}\gamma^\tau r_\tau+\gamma^HQ(x_H,\pi(x_H,\theta))|\theta]$
* For $H=0$, this is identical to the DDPG objective! 
$J(\theta)=\mathbb{E}_{x_0\sim\mu}[Q(x_0,\pi(x_0,\theta))]$

**Outlook: Alternative uncertainty propagation**
* Instead of using Monte Carlo rollouts to evaluate a policy, there are more refined ways to approximate the expected performance
    * Moment matching ($\to$ PILCO)
    * Variational inference

**What about unknown dynamics?**
*So far, have assumed a known (deterministic or stochastic) transition model $f$ and known reward $r$
* Natural approach if $f$ and $r$ are **unknown**:
    *  Start with initial policy $\pi$
    * Iterate for several episodes
        * Roll out policy $\pi$ to collect data
        * Learn a model for $f$, $r$ (and $Q$) from the collected data
        * Plan a new policy $\pi$ based on the estimated models

**How can we learn $f$ and $r$?**
* Key insight: due to the Markovian structure of the MDP, observed transitions and rewards are **(conditionally) independent**
$x_{t+1}\bot x_{1:t-1}|x_t,a_t\:;\:r_{t+1}\bot r_{1:t-1}|x_t,a_t$
* If we don’t know the dynamics & reward, can estimate them **off-policy** with **standard supervised learning techniques** from a replay buffer (data set)
$D=\{(x_i,a_i,r_i,x_{i+1})_i\}$

**Learning dynamics models $f$**
* For continuous state spaces, learning $f$ and $r$ is basically a regression problem
* Each experience $(x,a,r,x')$ provides a **labeled data point** $(z,y)$, with $z:=(x,a)$ as input and $y:=x'$ rsp. $r$ as label
* Below, we focus on **learning transition/dynamics models** $f$ (handling unknown rewards is analogous)
* In particular, we focus on challenges related to learning **probabilistic dynamics models** for
$x_{t+1}\sim f(x_t,a_t;\theta)$

**Example**
* Running example: **conditional Gaussian dynamics**
$x_{t+1}\sim\mathcal{N}(\mu(x_t,a_t;\theta),\Sigma(x_t,a_t;\theta))$
* Represent $\Sigma(x_t,a_t;\theta)$ via  lower triangular matrix $\Sigma(x_t,a_t;\theta)=C(x_t,a_t;\theta)C(x_t,a_t;\theta)^T$
* Advantage:
    * **Only needs $\frac{n(n+1)}{2}$ parameters**
    * Automatically guarantees **(semi)-definiteness**
    * Allows **reparametrization**: $x_{t+1}=\mu(x_t,a_t;\theta)+C(x_t,a_t;\theta)\epsilon\quad$  for $\epsilon\sim\mathcal{N}(0,I)$

**Learning with MAP estimation**
* First approach: obtain point estimate for $f$ via **MAP estimation** $\to$ need prior (regularizer) and likelihood
* Here, we focus on parametrizing $\mu(x,a,\theta)$ and $C(x,a,\theta)$ via a neural network
* Can obtain MAP estimate of weights $\theta=[w_{i,j}^{(k)}]$ via
$\hat{\theta}=argmin_\theta-logp(\theta)-\sum_{t=1:T}logN(x_{t+1}|\mu(x_t,a_t;\theta),\Sigma(x_t,a_t;\theta))$
* Can optimize using **stochatic gradient descent**

**Why MAP is not enough?**
* **Key pitfall in model-based RL**:
    * When planning over multiple time-steps ($H>1$), errors in the model estimate **compound**
    * This **compounding error is exploited** by planning algorithm (MPC, policy search)
    * This can result in **very poor performance**!
* This pitfall can be effectively remedied by **capturing uncertainty** in the estimated model, and **taking it into account in planning**
    $\to$ Separate epistemic and aleatoric uncertainty

**Reminder: Bayesian learning**
* Prior: $p(\theta)$
* Likelihood: $p(y_{1:n}|x_{1:n},\theta)\prod_{i=1}^n p(y_i|x_i,\theta)$
* Posterior: $p(\theta|x_{1:n},y_{1:n})=\frac{1}{Z}p(\theta)\prod_{i=1}^n p(y_i|x_i,\theta)$
where $Z=\int p(\theta)\prod_{i=1}^np(y_i|x_i,\theta)d\theta$
* Predictions: $p(y^*|x^*,x_{1:n},y_{1:n})=\int p(y^*|x^*,\theta)p(\theta|x_{1:n},y_{1:n})d\theta$

**Bayesian learning of dynamics models**
* Instead of obtaining a point estimate for $f$, we model a distribution over $f$. E.g., modeling $f$ as
    * Gaussian process
    * Bayesian neural network
* Finally get to use all the (approximate) inference techniques we learnt earlier!
    * Exact inference in GPs
    * Approximate inference in BNNs via variational inference, MCMC, dropout, ensembles,...

**Recall: Epistemic and aleatoric uncertainty**
* Suppose we obtain posterior distribution $P(f|D)$ for
$x_{t+1}\sim f(x_t,a_t)$
* Recall: we now have two forms of uncertainty
    * **Epistemic**: Uncertainty in $P(f|D)$
    * **Aleatoric**: Uncertainty in $P(x_{t+1}|f,x_t,a_t)$

**Example: Conditional Gaussians**
* Consider again our conditional Gaussian dynamics
$x_{t+1}\sim\mathcal{N}(\mu(x_t,a_t;\theta),\Sigma(x_t,a_t;\theta))$
* Most approximate inference techniques represent our **approximate posterior distribution** via
$P(x_{t+1}|f,x_t,a_t)\approx \frac{1}{M}\sum_{i=1:M}\mathcal{N}(\mu(x_t,a_t;\theta^{(i)}),\Sigma(x_t,a_t;\theta^{(i)}))$
* Hereby, the **epistemic uncertainty** is represented by the index of the mixture component $i$, and the **aleatoric uncertainty** by the variance within component $i$

**Separating epistemic and aleatoric uncertainty in planning**
* When planning, anticipate:
    * Dependent (consistent) behavior across $t$ acc. to $P(f|D)$
    * Independent randomness across $t$ acc. to $P(x_{t+1}|f,x_t,a_t)$
* Thus, our estimated expected performance becomes
$\hat{J}_H(a_{t:t+H-1})=\frac{1}{m}\sum_{i=1:m}\sum_{\tau=t:t+H-1}\gamma^{\tau-t}r_\tau(x_\tau(a_{t:\tau-1},\epsilon^{(i)}_{t:\tau-1},f^{(i)}),a_\tau)+\gamma^HV(x_{t+H})$
where $f^{(i)}\sim P(f|D)$ and 
$x_\tau:=x_\tau(a_{t:\tau-1},\epsilon_{t:\tau-1},f):=f(f(...f(x_t,a_t,\epsilon_t),a_{t+1},\epsilon_{t+1})...,a_{\tau-1},\epsilon_{\tau-1})$
for Gaussians, $x_{t+1}^{(i)}=\mu(x_t^{(i)},a_t^{(i)};\theta^{(j_i)})+C(x_t^{(i)},a_t^{(i)};\theta^{(j_i)})\epsilon_t^{(i)}$
where $j_i\sim Unif(\{1,...,m\})\quad \epsilon_t^{(i)}\sim \mathcal{N}(0,I)$

**Greedy exploitation for model-based RL**
* Start with empty data $D=\{\}$; prior $P(f)=P(f|\{\})$
* Iterate for several episodes
    * Plan a new policy $\pi$ to (approximately) maximize 
    $max_\pi\mathbb{E}_{f\sim P(\cdot|D)}J(\pi,f)$
    * Roll out policy $\pi$ to collect more data, add to $D$
    * Update posterior distribution $P(f|D)$

**PETS Algorithm [Chua, Calandra, McAllister, Levine 2018]**
* Uses an **ensemble of neural networks** each predicting
conditional Gaussian transition distributions
* **Trajectory sampling** is used to evaluate performance
* **MPC** used for planning

**How about exploration?**
* A key difference between RL and classical supervised learning is that the chosen actions affect the data we learn the models from
$\to$ **Exploration – Exploitation dilemma**
* How do we resolve this dilemma?
    * Adding **exploration noise** (e.g., Gaussian noise “dithering”)
    * **Thompson Sampling**
    * **Optimistic exploration**

**Thompson Sampling**
* We have already encountered Thompson / posterior sampling in context of Bayesian optimization
* The idea also applies to (model-based) RL
    * Start with empty data $D=\{\}$; prior $P(f)=P(f|\{\})$
    * Iterate for several episodes
        * Sample a model $f\sim P(f|D)$
        * Plan a new policy $\pi$ to (approximately) maximize 
        $max_\pi J(\pi,f)$
        * Roll out policy $\pi$ to collect more data, add to $D$
        * Update posterior distribution $P(f|D)$

**How about optimism?**
* Optimism is a central pillar for exploration in RL
* How about the model-based setting?
* Conceptionally, can consider a set $M(D)$ of models
that are **plausible** given data $D$
    * E.g., for conditional Gaussians
    $M(D)=\{f:f_i(x,a)\in \mu_i(x,a|D)\pm \beta\sigma_i(x,a|D)\forall x,a\}$

**Optimistic exploration**
* Start with empty data $D=\{\}$; prior $P(f)=P(f|\{\})$
* Iterate for several episodes
    * Plan a new policy $\pi$ to (approximately) maximize 
    $max_\pi max_{f\in M(D)}J(\pi,f)$
    * Roll out policy $\pi$ to collect more data, add to $D$
    * Update posterior distribution $P(f|D)$
* In general, the joint maximization over $\pi$ and $f$ is **very difficult**

**Optimistic Exploration in Deep Model-based RL: H-UCRL[Curi, Berkenkamp, Krause, NeurIPS 2020]**
$\pi_t^{H-UCRL}=argmax_{\pi(\cdot)}J(\tilde{f},\pi)\quad s.t. \tilde{f}(\mathbf{s},\mathbf{a})=\mu_{t-1}(\mathbf{s},\mathbf{a})+\beta_{t-1}\Sigma_{t-1}(\mathbf{s},\mathbf{a})\eta(\mathbf{s},\mathbf{a})$

**Illustration on Inverted Pendulum**
**Deep RL: Mujoco Half-Cheetah**
* H-UCRL outperforms Greedy & Thompson sampling Stronger effect for harder exploration tasks

**Action penalty effect**
* Small action penalty:
    * **Unrealistic behaviors allowed**
    * Exploration easy
    * Existing approaches work fine
* Large action penalty:
    * Avoids aggressive controls
    * **Exploration hard**
    * H-UCRL still finds good policies

**Outlook: Safe Exploration**
* In high-stakes applications, exploration is a dangerous proposition
* Need to guarantee **safety** (avoid unsafe states)
* How can we ensure this in case of unknown models?

**Planning with confidence bounds**
**Stylized task**
**Forwards-propagating uncertain, nonlinear GP dynamics [w Koller, Berkenkamp, Turchetta CDC ’18]**
* Thm: For conditional Gaussian dynamics, can overapproximate the reachable states w.p. $1-\delta$

**Challenges with long-term action dependencies**
* Can use confidence bounds for **certifying long-term safety!**

**Lyapunov functions**
* $x_{t+1}=f(x_t,\pi(x_t,\theta))$
* $V(x_{t+1})< V(x_t)\quad\forall x_t\in \mathcal{V}(c)\backslash\mathcal{V}(c_0)$
[A.M. Lyapunov 1892]

**Confidence-based Lyapunov analysis [Berkenkamp, Turchetta, Schoellig, K, NeurIPS 2017]**
* $Pr(V(x_{t+1})< V(x_t)\quad\forall x_t\in \mathcal{V}(c)\backslash\mathcal{V}(c_0))\geq1-\delta$
* Can also learn Lyapunov candidates via neural networks via reduction to classification
[Richards, Berkenkamp, K, CoRL ‘18]

**Safe learning-based MPC [Koller, Berkenkamp, Turchetta, K CDC ’18,’19]**
* Theorem (informally): Under suitable conditions can always guarantee that we are able to **return to the safe set**
[c.f. Wabersich & Zeilinger ‘18]

**Experiments [Koller, Berkenkamp, Turchetta, K CDC ’18, ’19]**

**What you need to know**
* Reinforcement learning = learning in MDPs
* Need to trade off **exploration and exploitation**
    * Epsilon-greedy
    * Thompson sampling
    * Optimistic exploration (Rmax, Optimistic Q-learning, H-UCRL, ...)
* Tabular model-based vs. model-free methods
* PAC-MDP results
* Scaling up by **approximating the value function** and using **parametric policies**
* Basic ideas for model-based Deep RL
* Can use **Bayesian learning** to utilize epistemic uncertainty during exploration

**You’ve learned a lot!**
Bayesian linear regression, Gaussian processes, variational inference, MCMC,
SGLD, Gibbs sampling, Kalman Filters, bandits, Bayesian optimization, Markov
Decision processes, value iteration, policy iteration, POMDPs, TD-learning, Q-learning, DQN, actor-critic methods, model-based deep reinforcement learning, PETS, H-UCRL

**Key concepts & notions**
* Bayesian learning
* Learning as inference
* Epistemic vs aleatoric uncertainty
* Score- and reparametrization gradient estimators
* POMDPs as belief-state MDPs
* Optimism in the face of uncertainty

**If you want to learn more**
* Other Courses
    * Deep Learning
    * Statistical Learning Theory
    * Guarantees for Machine Learning
    * Optimization for Data Science
    * Reliable and Interpretable AI
    * Computational Intelligence Lab
* Conference proceedings & Journals
    * AI: AAAI, IJCAI, JAIR
    * Machine Learning: ICML, NIPS, ICLR, AISTATS, JMLR, …
    * Robotics: ICRA, IROS, RSS, CoRL, IJRR, ...
* MSc. Thesis

