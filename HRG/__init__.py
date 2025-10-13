"""
HRG (Heterogeneous Resource Gathering) Environment Package

This package implements a multi-agent reinforcement learning environment
for heterogeneous resource collection tasks with different agent roles.
"""

from .env_hrg import HRGEnv, HRGConfig
from .env_hrg_ctde import HRGCTDEWrapper, create_hrg_ctde_env
from .env_hrg import create_hrg_env

__all__ = [
    'HRGEnv',
    'HRGConfig',
    'HRGCTDEWrapper',
    'create_hrg_ctde_env',
    'create_hrg_env'
]

__version__ = "1.0.1"