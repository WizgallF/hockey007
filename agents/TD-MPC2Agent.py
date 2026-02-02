from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import yaml
import os
import json
import random
import torch.optim as optim
from datetime import datetime
from agents.AgentBaseclass import Agent
from agents.networks.RainbowNetwork import RainbowNetwork
from agents.utils.RainbowUtils import ReplayBuffer, LinearSchedule


class TDMPC2Agent(Agent):
    def __init__(
            self,
            n_observations: int,
            n_actions: int,
            planning_horizon: int = 12,
            verbose = False
            ):

        # ------ load configs from "tdmpc_config.yaml" ------
        with open("configs/tdmpc_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        self.configs = config
        self.__dict__.update(config)


        self.n_observations = n_observations
        self.n_actions = n_actions
        self.verbose = verbose
        self.horizon = planning_horizon
        self.t = 0

        # ------ initialize neural networks ------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ------ optimizer ------
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR, betas=(self.ADAM_BETA_1, self.ADAM_BETA_2), eps=self.ADAM_EPS)

    def model_loss(
            self,
            reward_pred,
            q_pred,
            z_next_pred,
            next_state_pred,
            ):
        loss = 0.0
        return loss
        
    def policy_loss(
            self,
            z_pred):
        loss = 0.0
        return loss

        
    
    def act() -> int:
        """
        Given an observation, selects a discrete action. 
        Uses the epsilon-greedy strategy if network weights 
        are not noisy and greedy strategy is not used.
        ------------
        Parameters:
            env: The environment
            state: The current state
            i_episode: Number of episodes already seen
            statistics: The training statistics
            greedy: Whether greedy strategy for action selection is used
        -----------
        Return:
            Integer specifying the selected action.
        """
        pass
        

    def observe(
            self,
            state: np.ndarray, 
            action: int, 
            reward: int, 
            next_state: np.ndarray,
            terminated: bool
            ):
        """
        Saves observed transition in the replay buffer.
        ----------
        Parameters:
            state: Current state
            action: The action selected in current state
            reward: The reward given in current state for the selected action
            next_state: The next state of the environment
            terminated: Whether the episode has terminated after current transition

        """
        pass

    
    def update(self,
            ):
        """
        One training iteration of Ranbow Q-Learning with BATCH_SIZE samples drawn from the replay buffer.
        ----------
        Parameters:
            statistics: The training statistics
        """
        pass

        

    def save_dict(
            self,
            save_path: str = ""
            ):
        """
        Save the models state dict to specified path.
        ----------
        Parameter:
            save_path: The path where the model's state dictionary will be saved
        """
        saving_dir = os.path.join(save_path, self.MODEL_IDENTIFIER + ".pth")
        torch.save(self.policy_net.state_dict(), saving_dir)
    
    
    def load_dict(
            self,
            load_path: str = ""
            ):
        """
        Load the models state dict from specified path.
        ----------
        Parameter:
            load_path: The path from which the model's state dictionary will be loaded
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint)


    def save_experiment_config(
            self, 
            base_dir: str = ""
            ):
        """
        Creates a unique folder name and saves experiment configs to it.
        ----------
        Parameter:
            base_dir: The base directory for the experiment folder
        """
        # Create a unique folder name using timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{self.MODEL_IDENTIFIER}"
        experiment_path = os.path.join(base_dir, folder_name)
        
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save configs to experiment folder
        with open(f"{experiment_path}/config.yaml", 'w') as f:
            yaml.dump(self.configs, f)
            
        return experiment_path
    

    def print_config(self):
        """
        Prints the config to the terminal.
        """
        # indent=4 makes it look like a structured config file
        pretty_conf = json.dumps(self.configs, indent=4)
        print(f"Loading Agent: {self.MODEL_IDENTIFIER}")
        print(pretty_conf)
        
    

