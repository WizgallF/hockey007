from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

class Agent(ABC):
    @abstractmethod
    def act(
        obs: np.ndarray
        ) -> np.ndarray:
        """
        Docstring for act
        """
    
    @abstractmethod
    def observe(
        self,
        state, 
        action, 
        reward, 
        next_state):
        """
        Docstring for observe
        """

    @abstractmethod
    def update(
        epoch: int = None):
        """
        Docstring for train
        """

    @abstractmethod
    def save(
        save_path: str = ""
    ):
        """
        Docstring for save
        """
    
    @abstractmethod
    def load_params(
        load_path: str = ""):
        """
        Called in __init__ to initialize model
        """

    