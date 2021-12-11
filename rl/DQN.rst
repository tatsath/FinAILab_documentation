.. _DQN:



Deep Q-Learning
====

Q-learning may have the following drawbacks:
      * In cases where the state and action space are large, the optimal Q-value table quickly becomes computationally infeasible.
      * Q-learning may suffer from instability and divergence.
To address these shortcomings, we use ANNs to approximate Q-values. For example, if we use a function with parameter θ to calculate Q-values, we can label the Q-value function as Q(s,a;θ). The deep Q-learning algorithm approximates the Q-values by learning a set of weights, θ, of a multilayered deep Q-network that maps states to actions. The algorithm aims to greatly improve and stabilize the training procedure   of Q-learning through two innovative mechanisms:
Experience replay
Instead of running Q-learning on state-action pairs as they occur during simula‐ tion or actual experience, the algorithm stores the history of state, action, reward, and next state transitions that are experienced by the agent in one large replay memory. This can be referred to as a mini-batch of observations. During Q- learning updates, samples are drawn at random from the replay memory, and thus one sample could be used multiple times. Experience replay improves data efficiency, removes correlations in the observation sequences, and smooths over changes in the data distribution.
Periodically updated target
Q is optimized toward target values that are only periodically updated. The Q- network is cloned and kept frozen as the optimization targets every C step (C is a hyperparameter). This modification makes the training more stable as it over‐ comes the short-term oscillations. To learn the network parameters, the algo‐ rithm applies gradient descent9 to a loss function defined as the squared difference between the DQN’s estimate of the target and its estimate of the Q- value of the current state-action pair, Q(s,a:θ). The loss function is as follows:

L (θi) =  (r + γmax Q(s ′, a ′ ; θi–1) – Q(s, a; θi))2
a ′





    
The loss function is essentially a mean squared error (MSE) function, where
(r + γmaxa ′ Q(s , a ′ ; θi–1)) represents the target value and Q s, a; θi represents the
′
predicted value. θ are the weights of the network, which are computed when the loss function is minimized. Both the target and the current estimate depend on the set of weights, underlining the distinction from supervised learning, in which targets are fixed prior to training.
An example of the DQN for the trading example containing buy, sell, and hold actions is represented in Figure 9-6. Here, we provide the network only the state (s)  as input, and we receive Q-values for all possible actions (i.e., buy, sell, and hold) at once. 
Figure 9-6. DQN