import argparse
import os

import numpy as np

import train_agent
from rl_agent import Agent
from dynamics import simpleModel


def main(feed):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--pre_epochs', type=int, default=200)
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--expert_dir', type=str, default='FZ_demo.npz')
    parser.add_argument('--batch_num', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/pretrain", exist_ok=True)

    expert_data = np.load('expert_traj/' + args.expert_dir, allow_pickle=True)
    expert = expert_data

    sample = np.loadtxt("demo_mean/sample.csv", delimiter=",")
    env = simpleModel(sample)

    input_shape = (2,)
    action_space_dim = 2

    agent = Agent(input_shape=input_shape, action_space_dim=action_space_dim, max_epoch=args.epochs, batch_num=args.batch_num)

    train_agent.pretrain(env=env, agent=agent, pre_epochs=args.pre_epochs, gamma=args.gamma, expert=expert, max_step=args.max_step)

    agent.generator.load_model(path="checkpoints/pretrain/model_P_gen")

    train_agent.train_with_evaluate(
        env=env, feed=feed, agent=agent, epochs=args.epochs, gamma=args.gamma, max_step=args.max_step)

    agent.generator.session.close()


if __name__ == '__main__':
    feed = {
        "lr_schedule_epopch": [0, 1000],
        "gen_schedule": [1e-5, 1e-6],
        "stop_length": 200,
    }
    main(feed)
