from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

class Agent(ABC):
    @abstractmethod
    def act(
            self,
            env,
            state: np.ndarray,
            statistics: dict[str, list] = None
            ):
        """
        Given an observation, selects an action.
        ------------
        Parameters:
            env: The environment
            state: The current state
            statistics: The training statistics
        -----------
        Return:
            Selected action.
        """
    
    @abstractmethod
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

    @abstractmethod
    def update(
            self,
            statistics: dict[str, list] = None
            ):
        """
        One training iteration of Ranbow Q-Learning with BATCH_SIZE samples drawn from the replay buffer.
        ----------
        Parameters:
            statistics: The training statistics
        """

    @abstractmethod
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
    
    @abstractmethod
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

    