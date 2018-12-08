import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 train_environment, test_environment, num_tests, experience_replay,
                 agent, optimizer,
                 writer, write_frequency):
        # environments
        self.train_environment = train_environment
        self.test_environment = test_environment
        self.num_tests = num_tests
        self.experience_replay = experience_replay

        # agent and optimizer
        self.agent = agent
        self.optimizer = optimizer

        # writer
        self.writer = writer
        self.write_frequency = write_frequency

        self.action_dict = {
            0: [-1, 0, 0],
            1: [+1, 0, 0],
            2: [0, 1, 0],
            3: [0, 0, 0.8],
            4: [0, 0, 0]
        }

    def train_step(self, batch):
        loss = self.agent.loss_on_batch(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_performance(self):
        """just runs current agent in test environment until end
         and returns full episode reward
         """
        observation = self.test_environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = self.agent.act(observation)
            env_action = self.action_dict[action]
            observation, reward, done, _ = self.test_environment.step(env_action)
            episode_reward += reward
        return episode_reward

    def train(self, num_epochs, num_steps, batch_size):
        """main function of the class

        Train DQN agent with epsilon-greedy action selection. Epsilon linearly decay from 1.0 to 0.1
        if first half of training

        :param num_epochs: number of training epochs, agent will be tested at every epoch end
        :param num_steps: number of training steps per epoch
        :param batch_size:
        :return:
        """
        # test performance before training
        test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
        self.writer.add_scalar('test_reward', test_reward / self.num_tests, 0)

        observation = self.train_environment.reset()
        loss, mean_reward = 0.0, 0.0
        epsilon, decay = 1.0, 0.9 / (num_epochs / 2)

        # fill buffer
        for _ in tqdm(range(batch_size * 2)):
            action = np.random.randint(5)
            env_action = self.action_dict[action]
            new_observation, reward, done, _ = self.train_environment.step(env_action)
            self.experience_replay.push(observation, action, reward, done)

        for epoch in range(num_epochs):
            for step in tqdm(range(num_steps), desc='epoch_{}'.format(epoch+1), ncols=80):
                # add new experience to the buffer
                if np.random.rand() < epsilon:
                    action = self.agent.act(observation)
                else:
                    action = np.random.randint(5)
                env_action = self.action_dict[action]
                new_observation, reward, done, _ = self.train_environment.step(env_action)
                self.experience_replay.push(observation, action, reward, done)

                # sample some experience for training from the buffer
                batch = self.experience_replay.sample(batch_size)
                loss = self.train_step(batch)

                # update writer statistics
                loss += loss
                mean_reward += batch[2].mean()

                # write logs
                if (epoch * num_steps + step) % self.write_frequency == 0:
                    d = self.write_frequency  # 'd' stands for 'denominator'
                    log_step = (epoch * num_steps + step) // d
                    self.writer.add_scalar('loss', loss / d, log_step)
                    self.writer.add_scalar('batch_reward', mean_reward / d, log_step)
                    self.writer.add_scalar('epsilon', epsilon, log_step)
                    loss, mean_reward = 0.0, 0.0

                # update current observation
                if done:
                    observation = self.train_environment.reset()
                else:
                    observation = new_observation
            # test performance at the epoch end
            test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
            self.writer.add_scalar('test_reward', test_reward / self.num_tests, epoch + 1)
            # update target network and decrease epsilon
            self.agent.update_target()
            epsilon = max(0.1, epsilon - decay)
