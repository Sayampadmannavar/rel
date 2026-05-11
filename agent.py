import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:

    def __init__(self):

        self.state_size = 4
        self.action_size = 2

        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        model.add(
            Dense(
                24,
                input_dim=self.state_size,
                activation='relu'
            )
        )

        model.add(
            Dense(
                24,
                activation='relu'
            )
        )

        model.add(
            Dense(
                self.action_size,
                activation='linear'
            )
        )

        model.compile(
            loss='mse',
            optimizer=Adam(
                learning_rate=self.learning_rate
            )
        )

        return model

    def choose_action(self, state):

        if np.random.rand() <= self.epsilon:

            return random.randrange(
                self.action_size
            )

        q_values = self.model.predict(
            state,
            verbose=0
        )

        return np.argmax(q_values[0])

    def train(
        self,
        state,
        action,
        reward,
        next_state
    ):

        target = reward

        future_q = np.amax(
            self.model.predict(
                next_state,
                verbose=0
            )[0]
        )

        target = reward + (
            self.gamma * future_q
        )

        target_f = self.model.predict(
            state,
            verbose=0
        )

        target_f[0][action] = target

        self.model.fit(
            state,
            target_f,
            epochs=1,
            verbose=0
        )

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay