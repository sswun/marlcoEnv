# DEM Environment

## Overview

The DEM (Defense, Escort, and Movement) environment is a multi-agent reinforcement learning environment where agents must escort a VIP through dangerous territory while dealing with various threats.

## Quick Start

### Basic Usage

```python
from Env.DEM.env_dem import create_dem_env

# Create environment
env = create_dem_env(difficulty="easy", max_steps=100)

# Reset and get observations
obs = env.reset()
print(f"Number of agents: {len(obs)}")

# Take actions
actions = {agent_id: np.random.randint(0, 10) for agent_id in obs.keys()}
next_obs, rewards, done, info = env.step(actions)

# Get VIP information
vip_state = env.game_state.vip.get_state()
print(f"VIP HP: {vip_state['hp']}/{vip_state['max_hp']}")
print(f"VIP Position: {vip_state['pos']}")

env.close()
```

### CTDE Usage (for MARL algorithms like QMIX, VDN, etc.)

```python
from Env.DEM.env_dem_ctde import create_dem_ctde_env

# Create CTDE environment
ctde_env = create_dem_ctde_env(
    difficulty="normal",
    global_state_type="concat",  # Options: concat, mean, max, attention
    max_steps=200
)

# Reset and get observations and global state
obs = ctde_env.reset()
global_state = ctde_env.get_global_state()

print(f"Agents: {len(obs)}")
print(f"Global state shape: {global_state.shape}")

ctde_env.close()
```

## Environment Features

### Core Mechanics
- **VIP Escort**: Protect a VIP while helping them reach a target position
- **Dynamic Threats**: Enemies spawn adaptively based on VIP status
- **Role Emergence**: Agents naturally develop different strategies
- **Terrain Effects**: Rivers and forests affect movement and combat

### Actions (10 per agent)
- 0: Move Up
- 1: Move Down
- 2: Move Left
- 3: Move Right
- 4: Attack
- 5: Observe
- 6: Guard VIP
- 7: Communicate
- 8: Long Range Attack
- 9: Stay Still

### Difficulty Levels
- **Easy**: Smaller grid, fewer threats, more VIP HP
- **Normal**: Balanced gameplay
- **Hard**: More threats, aggressive spawning, time pressure

### Global State Types (CTDE mode)
- **concat**: Concatenated state of all agents and environment
- **mean**: Mean-pooled agent observations
- **max**: Max-pooled agent observations
- **attention**: Observation + attention weights

## Key Information

### Environment Info
```python
env_info = env.get_env_info()
# Returns: n_agents, obs_shape, n_actions, etc.
```

### VIP Status (Basic Environment)
```python
vip_state = env.game_state.vip.get_state()
# Returns: pos, target_pos, hp, max_hp, is_alive, etc.
```

### VIP Status (CTDE Environment)
```python
global_info = ctde_env.get_global_info()
# Returns comprehensive info including VIP, agents, threats, terrain
```

### Episode Statistics
```python
stats = ctde_env.get_stats()
# Returns detailed statistics including role emergence metrics
```

## Testing

Run the test suite to verify installation:

```bash
# Run comprehensive tests
python run_dem_tests.py

# Run tutorial examples
python test_tutorial_example.py
```

## Performance

- **Speed**: >7000 steps/second
- **Agents**: 3 agents by default (configurable)
- **Grid Size**: 10x12 (easy), 12x12 (normal), 12x12 (hard)
- **Max Steps**: 100 (easy), 200 (normal), 150 (hard)

## Files Structure

```
Env/DEM/
├── __init__.py          # Package initialization
├── core.py              # Core classes (Agent, VIP, Threat, etc.)
├── config.py            # Configuration management
├── env_dem.py           # Base environment implementation
├── env_dem_ctde.py      # CTDE wrapper for MARL algorithms
├── test_env.py          # Test suite
└── README.md            # This file
```

## Integration with RL Algorithms

The DEM environment is compatible with:
- QMIX (Q-Mixing)
- VDN (Value Decomposition Networks)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- Other CTDE (Centralized Training, Decentralized Execution) algorithms

### Data Format Example
```python
# For MARL algorithms
episode_data = {
    'obs': [obs],                    # List of observation dicts
    'actions': [0, 1, 2],            # Action indices per agent
    'rewards': [0.1, -0.05, 0.2],    # Rewards per agent
    'global_state': [global_state],  # Global state (CTDE mode)
    'done': [False],                 # Done flag
    'agent_ids': list(obs.keys())    # Agent identifiers
}
```

## Issues Fixed

1. **Import errors**: Fixed relative imports in all modules
2. **Method naming**: Fixed `update_cooldowns()` → `update_cooldown()`
3. **API consistency**: Ensured consistent interface between base and CTDE environments
4. **Tutorial compatibility**: Fixed Jupyter notebook examples to work with both base and CTDE environments

## Next Steps

- Experiment with different reward shaping strategies
- Develop specialized communication protocols
- Create custom terrain configurations
- Analyze emergent behaviors and role specialization
- Integrate with your favorite MARL algorithms