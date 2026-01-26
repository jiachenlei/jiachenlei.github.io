
####  Remaining questions (Updating) 
There're several questions that remain to be further explored. I'll answer them on-by-one once experiments are finished. 

- In the context of RF, what's the relationship between the curvature of the sampling trajectory and the "straightness" and speed uniformity of the forward ODE? A theoretical analysis

> Paper for reference: 
> - Minimizing Trajectory Curvature of ODE-based Generative Models: study the relationship between sampling curvature and forward process. It proposes that the curvature is determined by the coupling degree between noise $\epsilon$ and clean data $\bm{x}$ or, more simply, the way we sample the training data pair ($\epsilon$, $\bm{x}$). This is similar rationale of the immiscible diffusion paper which "pairs" $\bm{x}$ with the noise $\epsilon$ that is closest to $\bm{x}$ in Euclidean distance.
> - Accelerating Diffusion Training with Noise Assignment

- In the context of DMs, if we train DM (VP-SDE) by adopting the F-prediction in EDM and sample via the $\epsilon$\-prediction (thus the sampling amplifies the model prediction error), would this improve the model generation performance in comparison with the counterpart trained with $\epsilon$\-prediction (also sample with $\epsilon$\-prediction)? Moreover, would sampling with v-prediction improve the model generation performance even when parameterized with $\epsilon$\-prediction ?

- Can we transform RFs to DMs and vice versa? A theoretical analysis

### Other interesting topics
1. Connections between Loss weighting and time sampling schedule？
Takeaways:
- Setting loss weight w.r.t $\sigma(t)$ is equivalent to setting a time sampling schedule from the perspective of optimization.
- As suggested in EDM, it's more preferrable to decouple the role of loss weight and time sampling schedule. The loss weight balances effective loss values across time steps and time sampling schedule to stress training efforts within relevant time range.

*Question: what about EDM2 and the adaptive loss weighting used in ECT (pseudo-huber loss)*

2. EDM Done Right. Takeaways:
- Diffusion models trained with any framework can adopt any ODE samplers, even those that are different from the one defined by the forward SDE used during training.
- Time discretization schedule matters: allocate less at large t and more at small t.
- Design of the SDE matters: set $s(t)=1$ and $\sigma(t)=t$ benefit sampling.
- Model parameterization matters: predict mixture of $\bm{x}(0)$ and $\bm{\epsilon}$.
- Loss weights/Time step sampling schedule matters

3. Tweedie's formula and its relation to the score $\nabla_x\log p_t(\bm{x})$

Given the denoising score matching training objective:

$$
\theta^* = \argmin_{\theta}  \mathbb{E}_{t} \bigl\{ \lambda(t)\mathop{\mathbb{E}}_{\bm{x}(0)}\mathop{\mathbb{E}}_{\bm{x}(t)\mid\bm{x}(0)} \big[ || s_{\bm{\theta}}(\bm{x}(t), t) - \nabla_{\bm{x}(t)}\log p_t(\bm{x}(t)\mid\bm{x}(0)) ||^2_2 \bigr] \bigr\},
$$

Then, we derive the exact minimum of the above equation at time $t$. Decompose the norm, we have:
$$

$$