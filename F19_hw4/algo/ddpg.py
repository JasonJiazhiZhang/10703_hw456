import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy
from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001


class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev x`of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(1337)
        self.env = env
        self.outfile = outfile_name
        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)

        self.actor = ActorNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_ACTOR)
        self.critic = CriticNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_CRITIC)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.noise = EpsilonNormalActionNoise(0, 0.01, 0.2)

    def evaluate(self, num_episodes, if_save=True, ):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.actor.model.predict(s_t[None])[0]
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
#             if i < 9:
#                 plt.subplot(3, 3, i+1)
#                 s_vec = np.array(s_vec)
#                 pusher_vec = s_vec[:, :2]
#                 puck_vec = s_vec[:, 2:4]
#                 goal_vec = s_vec[:, 4:]
#                 plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
#                 plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
#                 plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
#                 plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
#                 plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
#                                  color='g' if success else 'r')
#                 plt.xlim([-1, 6])
#                 plt.ylim([-1, 6])
#                 if i == 0:
#                     plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
#                 if i == 8:
#                     # Comment out the line below to disable plotting.
#                     if if_save:
#                         time_now = datetime.datetime.now()
#                         figure_str = "figures_her/plot_" + str(time_now.hour) + "_" + str(time_now.minute) + ".png"
                        
#                         plt.savefig(figure_str)
#                     plt.clf()
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """
        train_rewards = []
        test_success_rate = []
        test_avg_rewards = []
        for i in range(num_episodes):
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            loss = 0
            store_states = [state]
            store_actions = []
            while not done:
                action = self.actor.model.predict(np.expand_dims(s_t, 0))[0]
                # add noise
                action = self.noise(action)
                next_state, reward, done, info = self.env.step(action)        
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state

                batch = self.replay_buffer.get_batch(BATCH_SIZE)
                batch_size = len(batch)
                
                states = np.array([line[0] for line in batch]).reshape(batch_size, -1)
                actions = np.array([line[1] for line in batch]).reshape(batch_size, -1)
                rewards = np.array([line[2] for line in batch]).reshape(batch_size, -1)
                next_states = np.array([line[3] for line in batch]).reshape(batch_size, -1)
                
                dones = np.array([line[4] for line in batch]).reshape(batch_size, -1)
                target_next_actions = self.actor.model_target.predict(next_states)
                target_next_q_values = self.critic.model_target.predict([next_states, target_next_actions])

                true_rewards = np.zeros_like(rewards)
                for j in range(len(batch)):
                    true_rewards[j] = rewards[j] if dones[j] else rewards[j] + GAMMA * target_next_q_values[j]

                history = self.critic.model.fit([states, actions], true_rewards, verbose=0)
                loss += np.mean(history.history['loss'])
                pred_actions = self.actor.model.predict(states)
                action_gradients = np.array(self.critic.gradients(states, pred_actions)).reshape(batch_size, -1)
                self.actor.train(states, action_gradients)
                self.actor.update_target()
                self.critic.update_target()

                total_reward += reward
                store_actions.append(action)
                store_states.append(state)
                s_t = np.array(state)
                step += 1

            if hindsight:
                # For HER, we also want to save the final next_state.
                self.add_hindsight_replay_experience(store_states,
                                                     store_actions)
            del store_states, store_actions
            store_states, store_actions = [], []
            train_rewards.append(total_reward)
            # Logging
#            
            print("Episode %d: Total reward = %d, TD loss = %.2f, Steps = %d; Info = %s, " % (i, total_reward,loss / step, step, info['done']))
            if i % 100 == 0:
                is_save = True if i % 1000 == 0 else False
                successes, mean_rewards, std_rewards = self.evaluate(100, is_save)
                print('Evaluation: success = %.2f; return = %.2f, std = %.2f' % (successes, mean_rewards, std_rewards))
                test_success_rate.append(successes)
                test_avg_rewards.append([mean_rewards, std_rewards])
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f, %.2f \n" % (successes, mean_rewards, std_rewards))
        base = "her_" if hindsight else ""
        np.save(base + 'train_rewards.npy', np.array(train_rewards))
        np.save(base + 'test_success_rate.npy', np.array(test_success_rate))
        np.save(base + 'test_avg_rewards.npy', np.array(test_avg_rewards))
                    

    def add_hindsight_replay_experience(self, c_states, c_actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        states = copy.deepcopy(c_states)
        actions = copy.deepcopy(c_actions)
        her_states, her_rewards = self.env.apply_hindsight(states)
        states, nextstates = her_states[:-1], her_states[1:]
        dones = [False for _ in range(len(actions)-1)] + [True]    
        for i in range(len(actions)):
            state, action, reward, next_state,done = states[i],actions[i],her_rewards[i],nextstates[i],dones[i]
            self.replay_buffer.add(state, action, reward, next_state,done)