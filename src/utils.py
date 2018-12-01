import gym
import numpy as np


class EnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames=4):
        super(EnvWrapper, self).__init__(env)
        self.n_frames = n_frames

        env.reset()
        env.env.viewer.window.dispatch_events()

        shape = (4, 96, 96)
        self.buffer = np.zeros(shape)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=shape,
            dtype=np.float32
        )

    def observation(self, observation):
        observation = np.mean(observation, axis=-1)  # color to gray-scale
        observation = 2.0 * (observation / 255.0) - 1.0  # [0, 255] -> [-1.0, 1.0]
        return observation

    def step(self, action):
        env_obs, reward, done, info = self.env.step(action)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = self.observation(env_obs)
        return self.buffer, reward, done, info

    def reset(self):
        reset_buffer = []
        self.observation(self.env.reset())
        for i in range(10):  # first 10 observations are useless
            obs, _, _, _ = self.env.step(self.env.action_space.sample())
            reset_buffer.append(self.observation(obs))
        # noinspection PyUnboundLocalVariable
        self.buffer = np.array(reset_buffer[-4:])
        return self.buffer
