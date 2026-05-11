import numpy as np

from environment import TrafficEnvironment
from agent import DQNAgent


env = TrafficEnvironment()

agent = DQNAgent()

env.start()

episodes = 5

for episode in range(episodes):

    state = env.reset()

    state = np.reshape(
        state,
        [1, 4]
    )

    done = False

    total_reward = 0

    print(f"\nEpisode {episode}")

    while not done:

        action = agent.choose_action(
            state
        )

        next_state, reward, done = env.step(
            action
        )

        next_state = np.reshape(
            next_state,
            [1, 4]
        )

        agent.train(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        total_reward += reward

        print(
            "State:",
            next_state,
            "Reward:",
            reward
        )

    print(
        f"Episode Reward: {total_reward}"
    )

agent.model.save(
    "models/traffic_dqn.h5"
)

print(
    "\nTraining Completed"
)