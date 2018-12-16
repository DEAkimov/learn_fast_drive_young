import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 train_environment, test_environment, num_tests, experience_replay,
                 agent, optimizer,
                 logdir, writer, write_frequency):
        # environments
        self.train_environment = train_environment
        self.test_environment = test_environment
        self.num_tests = num_tests
        self.experience_replay = experience_replay

        # agent and optimizer
        self.agent = agent
        self.optimizer = optimizer

        # writer
        self.logdir = logdir
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
        self.agent.train()
        loss = self.agent.loss_on_batch(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_performance(self):
        """just runs current agent in test environment until end
         and returns full episode reward
         """
        self.agent.eval()
        observation = self.test_environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = self.agent.act(observation)
            env_action = self.action_dict[action]
            observation, reward, done, _ = self.test_environment.step(env_action)
            episode_reward += reward
        self.agent.train()
        return episode_reward

    def test_reward(self, step):
        test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
        self.writer.add_scalar('test_reward', test_reward / self.num_tests, step + 137)

    def fill_buffer(self, batch_size):
        observation = self.train_environment.reset()
        for _ in tqdm(range(batch_size * 2)):
            action = np.random.randint(5)
            env_action = self.action_dict[action]
            new_observation, reward, done, _ = self.train_environment.step(env_action)
            self.experience_replay.push(observation, action, reward, done)
            if done:
                observation = self.train_environment.reset()
            else:
                observation = new_observation
        return observation

    def play_and_record(self, observation, epsilon):
        # makes single environment step and stores (s, a, r, s', done) in replay buffer
        self.agent.train()
        if np.random.rand() > epsilon:
            action = self.agent.act(observation)
        else:
            action = np.random.randint(5)
        env_action = self.action_dict[action]
        new_observation, reward, done, _ = self.train_environment.step(env_action)
        # push (s, a, r, done) before reset!
        self.experience_replay.push(observation, action, reward, done)

        if done:
            observation = self.train_environment.reset()
        else:
            observation = new_observation

        return observation, reward, done

    def write_logs(self, loss, mean_reward, epsilon, step):
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('batch_reward', mean_reward, step)
        self.writer.add_scalar('epsilon', epsilon, step)

    @staticmethod
    def flip_batch(batch):
        observations, actions, rewards, next_observations, is_done = batch
        # flip observations
        flipped_obs = observations[:, :, :, ::-1]
        observations = np.concatenate((observations, flipped_obs), axis=0)
        flipped_next_obs = next_observations[:, :, :, ::-1]
        next_observations = np.concatenate((next_observations, flipped_next_obs), axis=0)
        # flip actions
        flipped_actions = []
        for a in actions:
            if a == 0.0:
                flipped_actions.append(1.0)
            elif a == 1.0:
                flipped_actions.append(0.0)
            else:
                flipped_actions.append(a)
        actions = np.concatenate((actions, np.array(flipped_actions)), axis=0)
        # stack rewards and done
        rewards = np.concatenate((rewards, rewards), axis=0)
        is_done = np.concatenate((is_done, is_done), axis=0)
        return observations, actions, rewards, next_observations, is_done

    def train(self, num_epochs, num_steps, batch_size, save_freq):
        """main function of the class

        Train DQN agent with epsilon-greedy action selection. Epsilon linearly decay from 1.0 to 0.1
        if first half of training

        :param num_epochs: number of training epochs, agent will be tested at every epoch end
        :param num_steps: number of training steps per epoch
        :param batch_size:
        :param save_freq: save checkpoint once per several epochs
        :return:
        """
        # test performance before training
        self.test_reward(0)

        # train statistics
        episode_reward, episodes_done = 0, 0
        mean_loss, mean_reward = 0.0, 0.0
        # epsilon, decay = 1.0, 0.9 / (num_epochs / 2)
        epsilon, decay = 0, 0

        # fill buffer
        observation = self.fill_buffer(batch_size)

        for epoch in range(num_epochs):
            for step in tqdm(range(num_steps), desc='epoch_{}'.format(epoch+1), ncols=80):
                # add new experience to the buffer
                observation, reward, done = self.play_and_record(observation, epsilon)

                # sample some experience for training from the buffer
                batch = self.experience_replay.sample(batch_size)
                batch = self.flip_batch(batch)
                loss = self.train_step(batch)

                # update writer statistics
                episode_reward += reward
                mean_loss += loss
                mean_reward += batch[2].mean()

                # write logs
                if done:
                    episodes_done += 1
                    self.writer.add_scalar('train_episode_reward', episode_reward, episodes_done + 568)
                    episode_reward = 0

                if (epoch * num_steps + step) % self.write_frequency == 0:
                    d = self.write_frequency  # 'd' stands for 'denominator'
                    log_step = (epoch * num_steps + step) // d + 6858
                    self.write_logs(mean_loss / d, mean_reward / d, epsilon, log_step)
                    mean_loss, mean_reward = 0.0, 0.0

            # test performance at the epoch end
            self.test_reward(epoch + 1)
            # update target network and decrease epsilon
            self.agent.update_target()
            # epsilon = max(0.1, epsilon - decay)
            # save agent
            if (epoch + 1) % save_freq == 0:
                self.agent.save(self.logdir + 'epoch_{}.pth'.format(epoch))
