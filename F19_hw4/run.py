import gym
import envs
from algo.ddpg import DDPG


def main():
    env = gym.make('Pushing2D-v0')
    algo = DDPG(env, 'ddpg_log.txt')
    algo.train(50000, hindsight=False)


if __name__ == '__main__':
    main()
