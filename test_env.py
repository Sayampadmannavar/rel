from environment import TrafficEnvironment

env = TrafficEnvironment()

env.start()

state = env.reset()

print("Initial State:", state)

for step in range(20):

    action = step % 2

    next_state, reward, done = env.step(action)

    print(
        f"Step {step}"
    )

    print(
        "State:",
        next_state
    )

    print(
        "Reward:",
        reward
    )

    print("----------------")

print("Test Completed")