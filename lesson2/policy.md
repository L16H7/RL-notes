# Policies and Value Functions in Reinforcement Learning (An introduction)

This serves as an informal introduction to fundamental RL concepts. It took me months to grasp these concepts, so I hope this note will help you.

## What is a value function?
Whenever you see the term "value function," you need to ask, "Under what policy?" This may be confusing at first, so allow me to demonstrate.

Let's use the game of tic-tac-toe as an example.
|   |   |   |
|---|---|---|
| X | O |   |
| O | O | X |
| X |   | X |

Assuming it is O's turn to play, you might know that O is going to win if you are an avid tic-tac-toe player. So, for you, the probability of winning is 100%. But imagine someone who has never played the game before. They would play the next move randomly, so their probability of winning is 50%, and their probability of losing is 50%. The idea here is that the state of the game has different values depending on the players and their skill levels.

In RL terms, the values of the state are different under different policies.

$v_{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]$

Don't worry if you don't understand the formula yet. $G_t$ is the return, which is basically all the rewards we can get from the state.

In our tic-tac-toe example, there are only two possible next states:

1. O plays a move and loses, resulting in a reward of -1.

|   |   |   |
|---|---|---|
| X | O | O |
| O | O | X |
| X |   | X |


2. O plays a move and wins, resulting in a reward of 1.

|   |   |   |
|---|---|---|
| X | O |   |
| O | O | X |
| X | O | X |

Let's consider a random player (random policy). We can say the expected reward is 0.

$$
p(\text{move}_1) \cdot \text{reward}_1 + p(\text{move}_2) \cdot \text{reward}_2
$$

$$
0.5 \cdot -1 + 0.5 \cdot 1 = 0
$$

### Equation time
$v_{\pi}(s) = \mathbb{E}_{\pi} [G_t \mid S_t = s]$

$v_{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_{\pi}(s')]$

Since this is an introduction, I will simplify this rather than strictly defining the correct terms.

$\gamma v_{\pi}(s')$ is the discounted value of the next state. In this game, since this is the last move the players make, there will not be more game states. So, we put this as 0.

$\sum_{s', r} p(s', r \mid s, a)$ is for non-deterministic games. In this case, the game is deterministic. So, for the sake of simplicity, I will put this term as 1.

Thus, the equation becomes:

$v_{\pi} = \sum_{a} \pi(a | s) \cdot r$

Which is the same is how we calculated:

$v_{\pi} = p(\text{move}_1) \cdot \text{reward}_1 + p(\text{move}_2) \cdot \text{reward}_2$

### A better policy
Let's say a better player can play the winning move with a 90% probability. The value of the state:

|   |   |   |
|---|---|---|
| X | O |   |
| O | O | X |
| X |   | X |

will be greater than for a random player.

$v_{\pi_{better}} = \sum_{a} \pi_{better}(a | s) \cdot r$
$= p(\text{move}_1) \cdot \text{reward}_1 + p(\text{move}_2) \cdot \text{reward}_2$
$= 0.1 \cdot -1 + 0.9 \cdot 1$
$= 0.8$

The goal in RL can be thought of as improving the policy to maximize rewards.
