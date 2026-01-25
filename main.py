from Core import Core
import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
import argparse
import platform


#Core.play(environment='hockey')


def main ():
    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument ( '--train', default=False, action="store_true", help="Train an agent in a specified environment" )
    parser.add_argument ( '--agent', type=str, default="rainbow", help = "Specify the agent to use for training/evaluation")
    parser.add_argument ( '--env', type=str, default="Hockey-One-v0", help = "Specify the environment to use for training/evaluation")
    parser.add_argument ( '--base_dir', type=str, default="experiments", help = "Specify the directory used for experimentation")
    parser.add_argument ( '--int_agents', default=False, action="store_true", help="Save intermediate agents during training" )
    parser.add_argument ( '--verbose', default=False, action="store_true", help="Enable verbosity during training/evaluation" )
    
    args = parser.parse_args()
    core = Core()

    if args.train:
        core.train_agent(args.agent, args.env, args.base_dir, args.int_agents, args.verbose)




if __name__ == '__main__':
    main()