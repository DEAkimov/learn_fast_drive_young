import gym
import torch
from tqdm import trange

from src.utils import EnvWrapper
from src.q_network import Agent

if __name__ == '__main__':
    test_env = EnvWrapper(gym.make('CarRacing-v0'), 4, 1)
    action_dict = {
        0: [-1, 0, 0],
        1: [+1, 0, 0],
        2: [0, 1, 0],
        3: [0, 0, 0.8],
        4: [0, 0, 0]
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(device)
    checkpoint = 'logs/exp_1/epoch_9.pth'
    agent.load(checkpoint)
    agent.eval()
    n_episodes = 3
    total_reward = 0  # 640 in eval, 650 in train
    for episode in trange(n_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = test_env.step(action_dict[action])
            test_env.render()
            episode_reward += reward
        # print('done, episode_reward: {}'.format(episode_reward))
        total_reward += episode_reward
    print(total_reward / n_episodes)
