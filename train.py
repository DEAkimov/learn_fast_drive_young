import torch
from tensorboardX import SummaryWriter

from src.utils import create_envs
from src.experience_replay import ExperienceReplay
from src.trainer import Trainer
from src.q_network import Agent

if __name__ == '__main__':
    train_env, test_env = create_envs()
    exp_replay = ExperienceReplay(100_000, train_env.observation_space.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(device)
    optimizer = torch.optim.Adam(agent.parameters(), 0.0002)
    logdir = 'logs/'
    writer = SummaryWriter(logdir)
    trainer = Trainer(train_env, test_env, 1, exp_replay, agent, optimizer, logdir, writer, 20)

    # r = trainer.test_performance()
    # print(r)
    trainer.train(500, 1000, 32, 10)