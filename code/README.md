# Lunar Robot Search Algorithm
## Setup Environment
run:
```
pip -r install requirements.txt
```

## Execute Instructions
- For setting up steps, access `lunar_rover_env.py` and change the `STEPS` variable inside `main()`
```python
if __name__ == "__main__":
    env = LunarRoverEnv(render_mode="human")
    observation, info = env.reset(seed=7)
    STEPS = 500     # -> change this to preferred step count for test
    print("Observation keys:", observation.keys())
    print("Initial info:", info)

    for _ in range(STEPS):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"action={action}, reward={reward:.2f}, info={info}")
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
```

- To run the program:
```
python code/lunar_rover_env.py
```
