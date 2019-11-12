import os
import tensorflow as tf
import numpy as np
import gym
import copy
import collections

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

        self.cem_actions = np.zeros(self.plan_horizon, self.action_dim)

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

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        # TODO: write your code here
        raise NotImplementedError

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        # TODO: write your code here
        new_states = np.zeros_like(states)
        for i in range(states.shape[0]):
            new_states[i, :] = self.env.get_nxt_state(states[i,:], actions[i,:])
        return new_states

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
        raise NotImplementedError

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
        if not self.use_mpc and t % self.plan_horizon != 0:
            action = self.cem_actions[t % self.plan_horizon, :]
            if t + 1 % self.plan_horizon == 0:
                self.reset_std()
            return action

        for i in range(self.max_iters):
            action_sample_x = np.random.normal(self.mean_x, self.std_x, (self.popsize, self.plan_horizon))
            action_sample_y = np.random.normal(self.mean_y, self.std_y, (self.popsize, self.plan_horizon))

            states = np.tile(state, (self.popsize, 1))
            num_state = states.shape[0]
            actions = np.column_stack((action_sample_x, action_sample_y))

            cost = np.zeros(num_state)
            for t in range(self.plan_horizon):
                next_states = self.predict_next_state(states, actions) # popsize * plan_horizon
                for j in range(num_state):
                    cost[j] += self.obs_cost_fn(next_states[i, :])
                states = next_states

            top_index = np.argsort(cost)[:self.num_elites]
            top_action_x = action_sample_x[top_index, :]
            top_action_y = action_sample_y[top_index, :]
            self.mean_x = np.mean(top_action_x, axis=0)
            self.mean_y = np.mean(top_action_y, axis=0)
            self.std_x = np.std(top_action_x, axis=0)
            self.std_y = np.std(top_action_y, axis=0)

        if not self.use_mpc:
            self.cem_actions = np.column_stack((self.mean_x, self.mean_y))
            action = self.cem_actions[0, :]

        return action


        # TODO: write your code here
        # raise NotImplementedError

    # TODO: write any helper functions that you need
