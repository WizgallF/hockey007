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
    parser.add_argument ( '--train_self_play', default=False, action="store_true", help="Train an agent in a specified environment to play against past versions of itself" )
    parser.add_argument ( '--play', default=False, action="store_true", help="Let an agent play against the basic opponent" )
    parser.add_argument ( '--agent', type=str, default="rainbow", help = "Specify the agent to use for training/evaluation")
    parser.add_argument ( '--num_parallel_envs', type=int, default=1, help = "Number of parallel environments to use during training")

    parser.add_argument ( '--playagent_1_type', type=str, default="rainbow", help = "Specify the agent 1 to use for evaluation")
    parser.add_argument ( '--playagent1_path', type=str, default=None, help = "Path to the trained agent 1 for evaluation")
    parser.add_argument ( '--playagent1_config', type=str, default="/home/nils-klute/Documents/machine_learning/Reinforcement Learning/hockey007/experiments_rainbow/2026-02-02_11-59-12_rainbow noise sigma0=0.5/config.yaml", help = "Path to the config file for agent 1 (only for evaluation)")

    parser.add_argument ( '--playagent_2_type', type=str, default="basicopp", help = "Specify the agent 2 to use for evaluation")
    parser.add_argument ( '--playagent2_path', type=str, default="/home/nils-klute/Documents/machine_learning/Reinforcement Learning/hockey007/saved_agents/rainbow.pth", help = "Path to the trained agent 2 for evaluation")
    parser.add_argument ( '--playagent2_config', type=str, default=None, help = "Path to the config file for agent 2 (only for evaluation)")

    parser.add_argument ( '--pop_path', type=str, default="/home/stud217/hockey007/experiments_rainbow/fixed_opponent_pool", help = "Path to the config file for agent 2 (only for evaluation)")
    
    parser.add_argument ( '--env', type=str, default="Hockey-One-v0", help = "Specify the environment to use for training/evaluation")
    parser.add_argument ( '--base_dir', type=str, default="experiments", help = "Specify the directory used for experimentation")
    parser.add_argument ( '--int_agents', default=False, action="store_true", help="Save intermediate agents during training" )
    parser.add_argument ( '--verbose', default=False, action="store_true", help="Enable verbosity during training/evaluation" )
    
    args = parser.parse_args()
    core = Core()

    if args.train:
        core.train_agent(
            args.agent, 
            args.env, 
            args.base_dir, 
            args.int_agents, 
            args.verbose,
            num_parallel_envs = args.num_parallel_envs
            )

    elif args.train_self_play:
        core.train_agent_self_play(
            agent_name=args.agent, 
            env_name=args.env, 
            base_dir=args.base_dir, 
            save_intermediate_agents=args.int_agents, 
            verbose=args.verbose, 
            agent_load_path=args.playagent1_path,
            population_path=args.pop_path)

    elif args.play:        
        core.play(
            environment=args.env, 
            player1=args.playagent_1_type, 
            player2=args.playagent_2_type, 
            agent1_path=args.playagent1_path, 
            agent1_config=args.playagent1_config, 
            agent2_path=args.playagent2_path, 
            agent2_config=args.playagent2_config)




if __name__ == '__main__':
    main()
