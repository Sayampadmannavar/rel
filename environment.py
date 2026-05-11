import traci
import numpy as np
import time


class TrafficEnvironment:

    def __init__(self):

        self.sumo_cmd = [
            "sumo-gui",
            "-c",
            "sumo_config/simulation.sumocfg",
            "--start"
        ]

        self.max_steps = 50

    def start(self):

        traci.start(self.sumo_cmd)

    def reset(self):

        self.step_count = 0

        return self.get_state()

    def get_state(self):

        lanes = [
            "north_in_0",
            "south_in_0",
            "east_in_0",
            "west_in_0"
        ]

        state = []

        for lane in lanes:

            count = traci.lane.getLastStepVehicleNumber(
                lane
            )

            state.append(count)

        return np.array(state)

    def step(self, action):

        # ONLY move simulation forward
        # No traffic light control yet

        traci.simulationStep()

        time.sleep(0.5)

        self.step_count += 1

        next_state = self.get_state()

        reward = -sum(next_state)

        done = self.step_count >= self.max_steps

        return next_state, reward, done