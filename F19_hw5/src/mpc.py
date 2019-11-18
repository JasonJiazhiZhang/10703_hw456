import os
import tensorflow as tf
import numpy as np
import gym
import copy
import collections
import random

class Replay_Memory():
    def __init__(self, memory_size=200):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # Hint: you might find this useful:
        #       collections.deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.memory = collections.deque(maxlen=memory_size)

    def sample(self, k):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return random.sample(self.memory, k)

    def append(self, transition):
        # Appends transition to the memory.           
        self.memory.append(transition)
    
    def __len__(self):
        return len(self.memory)

class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """

        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
        :param use_mpc: Whether to use only the first action of a planned trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """
        self.env = env
        self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
        self.num_particles = num_particles
        self.plan_horizon = plan_horizon
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters

        self.mean_x = np.zeros(self.plan_horizon)
        self.mean_y = np.zeros(self.plan_horizon)
        self.std_x = 0.5 * np.ones(self.plan_horizon)
        self.std_y = 0.5 * np.ones(self.plan_horizon)
        
        self.memory = Replay_Memory()
        self.train_states = None
        self.train_actions = None
        self.train_next_states = None

        # TODO: write your code here
        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        # raise NotImplementedError

    def obs_cost_fn(self, state):
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

    def obs_cost_fn_batch(self, states):
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5  

        pusher_x, pusher_y = states[:, 0], states[:, 1]
        box_x, box_y = states[:, 2], states[:, 3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box.T))
        d_goal = np.sqrt(np.dot(box_goal, box_goal.T))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord


    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        # TODO: write your code here
        # new_states = np.zeros_like(states)
        # for i in range(states.shape[0]):
        #     inputs = np.concatenate((states[i,:], actions[i,:]), axis=0)
        #     inputs = np.expand_dims(inputs, axis=0)
        #     index = np.random.randint(self.num_nets)
        #     out_means, out_logvars = self.model.sess.run(
        #         [self.model.means[index], self.model.logvars[index]], 
        #         feed_dict={self.model.models[index].input: inputs}
        #     )
        #     # mean = out_means[index]
        #     # logvar = out_logvars[index]
        #     new_state = np.random.normal(out_means, np.sqrt(np.exp(out_logvars)))
        #     new_states[i, :] = new_state

        # return new_states

        inputs = np.column_stack((states, actions))
        index = np.random.randint(self.num_nets, size = states.shape[0])
        
        model_index = []
        for i in range(self.num_nets):
            model_index.append(np.argwhere(index == i).squeeze())
        
        out_means = np.zeros_like(states)
        out_logvars = np.zeros_like(states)
        
        for i in range(self.num_nets):
            means, logvars = self.model.sess.run(
                [self.model.means[i], self.model.logvars[i]],
                feed_dict = {self.model.models[i].input: inputs[model_index[i]]}
            )
            out_means[model_index[i]] = means
            out_logvars[model_index[i]] = logvars
        return np.random.normal(out_means, np.sqrt(np.exp(out_logvars)))
    
#         inputs = np.column_stack((states, actions))
#         index = np.random.randint(self.num_nets)
#         out_means, out_logvars = self.model.sess.run(
#             [self.model.means[index], self.model.logvars[index]],
#             feed_dict = {self.model.models[index].input: inputs}
#         )
#         return np.random.normal(out_means, np.sqrt(np.exp(out_logvars)))



    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        # TODO: write your code here
        new_states = np.zeros_like(states)
        for i in range(states.shape[0]):
            new_states[i, :] = self.env.get_nxt_state(states[i,:], actions[i,:])
        return new_states
#         return self.env.get_nxt_state(states, actions)

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        # TODO: write your code here


        # for i in range(len(obs_trajs)):
        #     if self.train_states is not None:
        #         self.train_states = np.append(self.train_states, obs_trajs[i][:-1, :-2], axis=0)
        #     else:
        #         self.train_states = np.copy(obs_trajs[i][:-1, :-2])

        #     if self.train_actions is not None:
        #         self.train_actions = np.append(self.train_actions, acs_trajs[i], axis=0)
        #     else:
        #         self.train_actions = np.copy(acs_trajs[i])

        #     if self.train_next_states is not None:
        #         self.train_next_states = np.append(self.train_next_states, obs_trajs[i][1:, :-2], axis=0)
        #     else:
        #         self.train_next_states = np.copy(obs_trajs[i][1:, :-2])

        for i in range(len(obs_trajs)):
            obs = copy.deepcopy(obs_trajs[i])
            act = copy.deepcopy(acs_trajs[i])
            self.memory.append([obs, act])

        sampled_exp = self.memory.sample(len(obs_trajs))   
        

        for obs_traj, acs_traj in sampled_exp:
#         for obs_traj, acs_traj in zip(obs_trajs, acs_trajs):

            if self.train_states is not None:
                self.train_states = np.append(self.train_states, obs_traj[:-1, :-2], axis=0)
            else:
                self.train_states = copy.deepcopy(obs_traj[:-1, :-2])

            if self.train_actions is not None:
                self.train_actions = np.append(self.train_actions, acs_traj, axis=0)
            else:
                self.train_actions = copy.deepcopy(acs_traj)

            if self.train_next_states is not None:
                self.train_next_states = np.append(self.train_next_states, obs_traj[1:, :-2], axis=0)
            else:
                self.train_next_states = copy.deepcopy(obs_traj[1:, :-2])            

        train_inputs = np.column_stack((self.train_states, self.train_actions))
        train_targets = copy.deepcopy(self.train_next_states)

        train_loss, train_rmse = self.model.train(train_inputs, train_targets, epochs=epochs)
        return np.mean(train_loss), np.mean(train_rmse)
        # for i in range(epochs):
        #     index = np.random.permutation(len(self.train_states))
        #     for batch_id in range(0, len(self.train_states), batch_size):
        #         batch_index = index[batch_id * batch_size: min((batch_id + 1) * batch_size, len(self.train_states))]
        #         model_inputs = []
        #         model_targets = []
        #         for inputs, targets in zip(train_inputs, train_targets):
        #             model_inputs.append(np.take(inputs, batch_index))
        #             model_targets.append(np.take(targets, batch_index))
        #         loss, rmse = self.model.train(model_inputs, model_targets)
        #         losses.append(np.mean(loss))
        #         rmses.append(np.mean(rmse))

                
    def reset_std(self):
        self.std_x = 0.5 * np.ones(self.plan_horizon)
        self.std_y = 0.5 * np.ones(self.plan_horizon)
        return

    def reset(self):
        # # TODO: write your code here
        # raise NotImplementedError
        self.mean_x = np.zeros(self.plan_horizon)
        self.mean_y = np.zeros(self.plan_horizon)
        self.std_x = 0.5 * np.ones(self.plan_horizon)
        self.std_y = 0.5 * np.ones(self.plan_horizon)
        return

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        if self.use_random_optimizer:
            self.reset()
            
        if not self.use_mpc and t % self.plan_horizon != 0:
            action = self.cem_actions[t % self.plan_horizon, :]
            if t + 1 % self.plan_horizon == 0:
                self.reset_std()
            return action

        iters = self.max_iters if not self.use_random_optimizer else 1
        for i in range(iters):
            action_sample_x = np.random.normal(self.mean_x, self.std_x, (self.popsize, self.plan_horizon))
            action_sample_x_particle = np.repeat(action_sample_x, self.num_particles, axis=0)
            action_sample_y = np.random.normal(self.mean_y, self.std_y, (self.popsize, self.plan_horizon))
            action_sample_y_particle = np.repeat(action_sample_y, self.num_particles, axis=0)

            self.goal = state[-2:]
            state_without_goal = copy.deepcopy(state)
            state_without_goal = state_without_goal[:-2]
            states = np.tile(state_without_goal, (self.num_particles * self.popsize, 1))
 
            num_state = states.shape[0]
            cost = np.zeros(num_state)
                     
            for t in range(self.plan_horizon):
                actions = np.column_stack((action_sample_x_particle[:, t], action_sample_y_particle[:, t]))
                next_states = self.predict_next_state(states, actions) # [popsize*num_particles, plan_horizon]
                for j in range(num_state):
                    cost[j] += self.obs_cost_fn(next_states[j, :])
                #cost += self.obs_cost_fn_batch(next_states)
                states = next_states
            
            cost = np.reshape(cost, (self.popsize, self.num_particles))
            # cost_particles = []
            # for i in range(self.popsize):
            #     cost_particles.append(np.mean(cost[i * self.num_particles : (i + 1) * self.num_particles]))
            cost_particles = np.mean(cost, axis=1)
            
            if self.use_random_optimizer:
                top_index = np.argsort(cost_particles)[0]
                self.mean_x = action_sample_x[top_index, :]
                self.mean_y = action_sample_y[top_index, :]
            else:
                top_index = np.argsort(cost_particles)[:self.num_elites]
                top_action_x = action_sample_x[top_index, :]
                top_action_y = action_sample_y[top_index, :]
                self.mean_x = np.mean(top_action_x, axis=0)
                self.mean_y = np.mean(top_action_y, axis=0)
                self.std_x = np.std(top_action_x, axis=0)
                self.std_y = np.std(top_action_y, axis=0)

        if not self.use_mpc:
            self.cem_actions = np.column_stack((self.mean_x, self.mean_y))
            action = self.cem_actions[0, :]
        else:
            action = np.array((self.mean_x[0], self.mean_y[0]))
            self.mean_x = np.append(self.mean_x[1:], 0)
            self.mean_y = np.append(self.mean_y[1:], 0)
            self.reset_std()
        return action


        # TODO: write your code here
        # raise NotImplementedError

    # TODO: write any helper functions that you need
