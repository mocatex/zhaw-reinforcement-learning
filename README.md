# RL Frozen Lake Report
geifab01 / feuchmor

## First-Visit and Multiple-Visit
We implemented both versions for the Monte Carlo Algorithm. For the smaller maps, the difference between these methods isn't that noticable.
We chose the First-Visit method becuase less noisy und more reliable.

## Monte Carlo Random
**Parameters:**
|5x5|11x11|
|<ul>
<li>episodes=10000</li>
<li>alpha=0.1</li>
<li>gamma=0.9</li>
<li>first_visit=True</li>
<li>greedy=False</li>
</ul>|<ul>
<li>episodes=30000</li>
<li>alpha=0.1</li>
<li>gamma=0.9</li>
<li>first_visit=True</li>
<li>greedy=False</li>
</ul>|
|<img width="594" height="495" alt="grafik" src="https://github.com/user-attachments/assets/ac59a1de-c73f-4e52-8b33-b6817a95a0d6" />|<img width="594" height="495" alt="grafik" src="https://github.com/user-attachments/assets/1cdade24-c91a-463e-a676-df7bae5e2c5d" />|

## Monte Carlo Incremental (greedy)
In this implementation the `greedy` parameter has to be `True`
Additionally, the parametes `epsilon` and `epsilon_decay` can be set.

We start with a high epsilon for a high exploration at the start.
The closer `epsilon_decay` is to 1, the longer the agent will randomly choose acitions and gradualy switch more to exploitation of the learnt value function.
For larger maps (11x11), it is feasible that the agent explores for a longer time (larger `epsilon_decay`).

|5x5|11x11|
|<ul>
<li>episodes=1000</li>
<li>alpha=0.1</li>
<li>gamma=0.9</li>
<li>epsilon=0.9</li>
<li>epsilon_decay=0.99</li>
<li>first_visit=True</li>
<li>greedy=True</li>
</ul>|<ul>
<li>episodes=12000</li>
<li>alpha=0.1</li>
<li>gamma=0.9</li>
<li>epsilon=1.0</li>
<li>epsilon_decay=0.999</li>
<li>first_visit=True</li>
<li>greedy=False</li>
</ul>|

## Q-Learning
