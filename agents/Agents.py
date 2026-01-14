from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

class Agent(ABC):
    @abstractmethod
    def act(
        state: np.ndarray
        ) -> np.ndarray:
        """
        Docstring for act
        """
    
    @abstractmethod
    def observe(
        state: np.ndarray, 
        action: np.ndarray, 
        reward: int, 
        done: bool):
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

    