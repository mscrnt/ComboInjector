# ComboInjector

A utility class for injecting combos, special moves, and basic actions in a fighting game environment. 
Currently tailored for _Street Fighter III_ (sfiii) with **multi-discrete** action spaces.

## Overview

`ComboInjector` manages:

- **Characters** and their special moves, each with probabilities.
- A method to **sample** new actions or combos for one or more agents.
- **Queued** sequences of moves (combos) that it executes one step at a time.
- Probability-based selection among jump, basic single-step, combos, or movement patterns.

## Key Features

- **Configurable** probabilities for each action category (jump / basic / combos / movement).
- **Frame skip** logic for charge and hold-based combos.
- **Reset** method to assign characters and super arts to each agent.
- **Multi-agent** friendly: Each agent maintains its own queue of moves.

## Usage

```
from ComboInjector import ComboInjector

# 1) Instantiate the class
injector = ComboInjector(environment_name='sfiii',
                         mode='multi_discrete',
                         frame_skip=4)

# 2) Reset with specific characters and super arts for each agent
# Suppose we have 2 players:
characters = ['Alex', 'Gouki']
super_arts = [1, 2]  # super art indices
injector.reset(characters, super_arts)

# 3) Sampling actions
# We can sample with default or custom probabilities
actions = injector.sample(prob_jump=0.05,
                          prob_basic=0.25,
                          prob_combo=0.40,
                          prob_cancel=0.20,
                          prob_movement=0.35)

# 'actions' is a dict with 'discrete' indices and 'multi_discrete' arrays for each agent
print("Discrete action for agent_0:", actions['discrete']['agent_0'])
print("Multi-discrete array for agent_0:", actions['multi_discrete']['agent_0'])

# 4) Next time step
actions_next = injector.sample()
# If the agent has a queued combo, the next step of that combo is popped automatically.

```

