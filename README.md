# ComboInjector
Action sampling for fighting games enabling special move sequences. Designed for use in reinforcement learning.

Use:
```
injector = ComboInjector()
injector.reset(characters=['Ken'], super_arts=[3])
action = injector.sample()

{'discrete': {'agent_0': 80,}, 'multi_discrete': {'agent_0': [8, 0]}}
```
