from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.spaces import Discrete
from rllab.envs.base import Step
import numpy as np


class RandomBanditEnv(Env):
    def __init__(self, k=5, n=10, goal=None):  # Can set goal to test adaptation.
        self._goal = goal
        self._k = k
        self._n = n
        self._t = 0

    @property
    def observation_space(self):
        obs_dim = 3 + self._k
        return Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    @property
    def action_space(self):
        return Discrete(n=self._k)

    def sample_goals(self, num_goals):
        return np.random.uniform(0.0, 1.0, size=(num_goals, self._k, ))

    def reset(self, reset_args=None):
        goal = reset_args
        if goal is not None:
            self._goal = goal
        elif self._goal is None:
            self._goal = np.random.uniform(0.0, 1.0, size=(self._k,))
        self._state = np.array([0., 0., 0.])
        self._state = np.concatenate((self._state, np.zeros(self._k)))
        self._t = 0
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._t += 1
        reward = 0.
        if np.random.uniform() < self._goal[action]:
            reward = 1.
        done = False
        if self._t >= self._n:
            done = True
        action_one_hot = np.zeros(self._k)
        action_one_hot[action] = 1.
        self._state = np.array([0., reward, done])
        self._state = np.concatenate((self._state, action_one_hot))
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done, goal=self._goal)

    def render(self):
        print('current state:', self._state)
