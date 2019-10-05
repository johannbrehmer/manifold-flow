# manifold-flow

**J. Brehmer and K. Cranmer 2019**

Experiments with flows on manifolds.

## Outline

We study ways to calculate the likelihood on a `d`-dimensional manifold in `n`-dimensional data space:
- Flow: the usual in `n` dimensions.
- PIE: a flow in `n` dimensions, but the base density factorizes in a usual unit Gaussian in `d` dimensions
times very narrow Gaussian in `n-d` dimensions. The likelihood calculated in this way is still defined over the
full data space. In the generative direction, the `n-d` dimensions are fixed to zero.   
- Exact manifold flow: a flow that maps a `d`-dimensional base density to `n`-dimensional target space. The exact likelihood
in `d`-dimensional space can be calculated (for instance for evaluation purposes). For training in realistic problems
this is probably too expensive.
- Approximate manifold flow: a flow that maps a `d`-dimensional base density to `n`-dimensional target space. Instead
of the exact likelihood in `d`-dimensional space, we calculate the likelihood up to a normalizing factor by projecting
a data point on to the manifold and then calculating the `n`-dimensional likelihood (as in PIE) for this point.

Questions:
- Is this approximate likelihood in the manifold flow approach indeed proportional to the exact likelihood?
Does the proportionality vary a lot with training?
- How slow is the exact manifold flow evaluation? Maybe it's good enough for training on small problems?
- What's the best way to find the true dimension of the submanifold? Comparing likelihoods with different dimensionality
is a bit weird (apples vs oranges)
- What happens when we go to disjoint manifolds, potentially with different dimensions? It's weird to mix likelihoods
of different dimensionality during training...

Experiments:
- Simulate data from a tractable density on a manifold times noise in the orthogonal directions. Compare the true log
likelihood of the data generated from the different models.
- Compare generated data after training on MNIST or CIFAR or ImageNet... but are there any good metrics?
- Scientific (likelihood-free) inference example: can we train likelihoods this way and plug them into a simple
inference engine and compare MMD?



## Acknowledgements

The code is strongly based on [the Neural Spline Flow code base](https://github.com/bayesiains/nsf) by C. Durkan,
A. Bekasov, I. Murray, and G. Papamakarios, see [1906.04032](https://arxiv.org/abs/1906.04032) for their paper.
