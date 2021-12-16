.. _ML_Reinforcement:


Reinforcement Learning
----------------------

   Learning from experiences, and the associated rewards or punishments,
   is the core concept behind *reinforcement learning* (RL). It is about
   taking suitable actions to maximize reward in a particular situation.
   The learning system, called an *agent*, can observe the environment,
   select and perform actions, and receive rewards (or penal‐ ties in
   the form of negative rewards) in return, as shown in `Figure
   1-6 <#_bookmark52>`__.

   Reinforcement learning differs from supervised learning in this way:
   In supervised learning, the training data has the answer key, so the
   model is trained with the correct answers available. In reinforcement
   learning, there is no explicit answer. The learning

   .. image:: ../_static/img/fig1.6.png

   system (agent) decides what to do to perform the given task
   and learns whether that was a correct action based on the reward. The
   algorithm determines the answer key through its experience.

   *Figure 1-6. Reinforcement learning*

   The steps of the reinforcement learning are as follows:

1. First, the agent interacts with the environment by performing an
   action.

2. Then the agent receives a reward based on the action it performed.

3. Based on the reward, the agent receives an observation and
   understands whether the action was good or bad. If the action was
   good—that is, if the agent received a positive reward—then the agent
   will prefer performing that action. If the reward was less favorable,
   the agent will try performing another action to receive a posi‐ tive
   reward. It is basically a trial-and-error learning process.