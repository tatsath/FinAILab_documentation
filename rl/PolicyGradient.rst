.. _PolicyGradient:



Policy Gradient
================

| Policy gradient is a policy-based method in which we learn a policy function, π, which is a direct map from each state to the best corresponding action at that state. It is a more straightforward approach than the value-based method, without the need for a Q-value function.
| Policy gradient methods learn the policy directly with a parameterized function respect to θ, π(a|s;θ). This function can be a complex function and might require a sophisticated model. In policy gradient methods, we use ANNs to map state to action because they are efficient at learning complex functions. The loss function of the ANN is the opposite of the expected return (cumulative future rewards).
|
| The objective function of the policy gradient method can be defined as:
| J (θ) = Vπθ (S1) = πθ V 1
| where θ represents a set of weights of the ANN that maps states to actions. The idea here is to maximize the objective function and compute the weights (θ) of the ANN.
| Since this is a maximization problem, we optimize the policy by taking the gradient ascent (as opposed to gradient descent, which is used to minimize the loss function), with the partial derivative of the objective with respect to the policy parameter θ:
| θ ← θ + ∂ J (θ)
|
| Using gradient ascent, we can find the best θ that produces the highest return. Com‐ puting the gradient numerically can be done by perturbing θ by a small amount ε in the kth dimension or by using an analytical approach for deriving the gradient.