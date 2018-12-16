import gym
import numpy as np


class EnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_frames, frame_skip):
        super(EnvWrapper, self).__init__(env)
        self.n_frames = n_frames
        self.frame_skip = frame_skip

        env.reset()
        env.env.viewer.window.dispatch_events()

        shape = (4, 84, 84)
        self.buffer = np.zeros(shape)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=shape,
            dtype=np.float32
        )

    @staticmethod
    def on_grass(observation):
        """Accept raw observation"""
        return observation[60:-20, 40:56, 1].mean() > 165.0

    def observation(self, observation):
        # Gray = .299R + .587G + .114B
        observation = 2.0 * (observation / 255.0) - 1.0  # [0, 255] -> [-1.0, 1.0]
        observation = np.average(observation, axis=-1, weights=[0.299, 0.587, 0.114])
        return observation[:84, 6:-6]

    def skip(self, action):
        reward = 0
        for i in range(self.frame_skip):
            obs, r, done, info = self.env.step(action)
            reward += r
            if done:
                break
        # noinspection PyUnboundLocalVariable
        return obs, reward, done, info

    def step(self, action):
        step_frames = []
        reward = 0
        for i in range(self.n_frames):
            env_obs, r, done, info = self.skip(action)
            step_frames.append(self.observation(env_obs))
            reward += r
            if done:
                for j in range(self.n_frames - i - 1):
                    step_frames.append(step_frames[-1])
                # self.reset()
                break
        # noinspection PyUnboundLocalVariable
        if self.on_grass(env_obs):
            reward -= 0.1
        return np.array(step_frames), reward, done, info

    def reset(self):
        self.env.reset()
        for i in range(4):
            obs, _, _, _ = self.step([0, 1, 0])
        # noinspection PyUnboundLocalVariable
        return obs


def create_envs(n_frames=4, frame_skip=1):
    train_env = EnvWrapper(gym.make('CarRacing-v0'), n_frames, frame_skip)
    test_env = EnvWrapper(gym.make('CarRacing-v0'), n_frames, frame_skip)
    return train_env, test_env
