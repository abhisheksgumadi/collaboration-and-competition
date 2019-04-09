
## Learning Algorithm

The agent is trained with the [Multi Agent DDPG](https://arxiv.org/abs/1706.02275). The full algorithm is described in the METHODS section of the paper.

The information flow in a MADDPG algorithm is depeicted below (Screenshot from the paper)

![MADDPG](./images/maddpg.png)

- We initialize the replay memory `D` to some capacity `N`.
- We initialize the local Q-Network (which should approximate the true action-value function Q) with random weights (using PyTorch default weight initialization for any module).
- We copy those generated weights to the target Q-Network (which shall be updated every some defined number of iterations).
- We train the agent for some episodes and for some maximum number of time-steps (`max_t`) in each episode, unless it terminates earlier (e.g. by encountering a terminal state).
- The training loop is composed out of two steps: sampling and learning.
- In the sampling step, the agent follows a behavioral (epsilon-greedy) policy, which given a state `s` returns an action `a`. It executes this action `a` and observes reward `r` and next state `s'`. The experince tuple (`s`, `a`, `r`, `s'`, `d`), where `d` is a boolean flag which denotes whether the episode will terminate in state `s'`, is stored in the replay buffer D.
- In the learning step, the agent samples uniformly at random a mini-batch of experience tuples from `D`.
- It uses these experience tuples to compute the targets by following a target (greedy) policy. The mean squared error between the target and expected action-values is computed and this TD-error is backpropagated in the *local* Q-Network so to update the weights using one step of SGD.
- Next, we update the *target* Q-Network weights by making a copy of the current weights of the local Q-Network.

**Architecture of Neural Network for the Q-function**

- input size = 37
- output size = 4
- 2 hidden layers and one output layer
- each hidden layer has 64 hidden units and is followed by a `ReLU` activation layer
- output layer is followed by a linear activation layer

**Hyperparameters**

```
BUFFER_SIZE = int(1e5) # capacity N of replay memory D
BATCH_SIZE = 64 # batch-size
GAMMA = 0.99 # discount factor
TAU = 1e-3 # soft-update factor (when updating the target network weights) 
LR = 5e-4 # learning rate for SGD
UPDATE_EVERY = 5 # when to update the local network weights
optimizer = ADAM (all hyper-parameters are fixed, except learning rate changed - see above)
seed = 0 # for reproducibility
```

## Plot of Rewards

![Scores](./results/scores_training.png)

```
Episode 100	Average Score: 1.05
Episode 200	Average Score: 4.62
Episode 300	Average Score: 7.12
Episode 400	Average Score: 9.40
Episode 497	Average Score: 13.00
Environment solved in 397 episodes!	Average Score: 13.00
```

## Observations/Issues

* Agent seems to get stuck if it collects all the yellow bananas in its near neighbourhood and is unable to escape the repetitive sequence of "back-and-forth" actions.

![Issue](./results/issue_blocked.gif)

* Keeping all other paramters fixed, `UPDATE_EVERY` was increased from 4 to 5 and yielded fewer number of episodes to solve the task. If it is set to 1, then the algorithm takes a long time to train (i.e. increasing the avg reward over the last 100 consecutive episodes) and does not seem to converge soon.

## Ideas for Future Work


- [Double DQN](https://arxiv.org/abs/1509.06461)
  - The authors show that DQN overestimates the action-values and it may harm performance in practice. Furthermore they show how double Q-learning extended to deep neural networks can prevent such overestimation
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
  - The motivation behind trying this algorithm is that its  main benefit is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm. Thus an experiment may be done where the same agent is used to interact with a similar environment as the Banana-environment. 
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
  - The idea behind using these technique for sampling from the replay buffer is that not all experiences are equal, some are more important than others in terms of reward, so naturally the agent should at least prioritize between the different experiences.
- **Learning from Pixels:** Solve the same task given that the states are not 37 values but 2D images composed of many pixels. Naturally a different DNN architecture has to be used for the Q-Network, such as a CNN. Furthermore, all the above algorithms and techniques may also improve performance.
