from typing import List, Dict
import gym
from gym import Env
import numpy as np


class HalfCheetah_Friction_Variance_Wrapper(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.initial_friction = self.env.model.geom_friction[5][0]  # Foot friction

    def reset_friction(self, magnitude: float) -> None:
        self.env.model.geom_friction[5][0] = magnitude * self.initial_friction


class HalfCheetah_Mass_Variance_Wrapper(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.initial_foot_mass = self.env.model.body_mass[4]

    def reset_friction(self, magnitude: float) -> None:
        self.env.model.body_mass[4] = magnitude * self.initial_foot_mass
