# Multi-Agent Reinforcement Learning Environment Collection

## 📖 Introduction

This folder contains a collection of environments designed specifically for Multi-Agent Reinforcement Learning (MARL). All environments follow a unified interface specification and support Centralized Training with Decentralized Execution (CTDE) algorithms such as QMIX, VDN, and MADDPG.

## 🎯 System Requirements

- **Operating System**: Ubuntu 20.04+ (Ubuntu 22.04/24.04 recommended)
- **Python Version**: Python 3.8+
- **Core Dependencies**: NumPy, Gymnasium, Pygame, Matplotlib

## 🏗️ Environment List

### 1. CM (Collaborative Moving) Environment

**Description**:  
A simple yet effective multi-agent collaboration environment where agents must cooperate to push a 2×2 box to a designated target area.

**Key Features**:
- 🤝 **Cooperation Mechanism**: Multiple agents push the box from different sides, with success rate increasing with cooperation
- 🎮 **Simple Actions**: 5 discrete actions (stay, move up/down/left/right)
- 🏆 **Team Rewards**: All agents receive the same team reward
- ⚙️ **Configurable Difficulty**: Supports easy, normal, and hard difficulty levels
- 👥 **Agent Count**: 2-4 agents (configurable)
- 📐 **Grid Size**: 7×7 grid (configurable)

**Key Parameters**:
- Action Space: 5 discrete actions
- Observation Space: 6 + 2×(n_agents-1) dimensional vector
- Max Steps: 50-100 steps (difficulty-dependent)

**Files**:
- `env_cm.py`: Base environment implementation
- `env_cm_ctde.py`: CTDE-compatible version
- `core.py`: Core class definitions
- `config.py`: Configuration management
- `renderer.py`: Visualization renderer

---

### 2. DEM (Dynamic Escort Mission) Environment

**Description**:  
A multi-agent reinforcement learning environment where agents dynamically form roles to escort a VIP through dangerous territory while dealing with various threats.

**Key Features**:
- 🎭 **Role Emergence**: Agents naturally form roles like defenders, guardians, and scouts
- 🗺️ **Complex Terrain**: Rivers and forests affect movement and combat
- 🎯 **Dynamic Threats**: Enemies spawn adaptively based on VIP status
- 💬 **Communication**: Supports inter-agent information exchange
- 🏥 **VIP Protection**: Core objective is to safely escort VIP to destination
- 👥 **Agent Count**: 3 special forces agents (configurable)
- 📐 **Grid Size**: 10×12 or 12×12 (difficulty-dependent)

**Key Parameters**:
- Action Space: 10 discrete actions (move, attack, observe, guard, communicate, etc.)
- Observation Space: ~60 dimensional vector (self state, VIP state, teammate state, threat state)
- Global State Dimension: 41 dimensions
- Max Steps: 100-200 steps (difficulty-dependent)

**Files**:
- `env_dem.py`: Base environment implementation
- `env_dem_ctde.py`: CTDE-compatible version
- `core.py`: Core classes (Agent, VIP, Threat, etc.)
- `config.py`: Configuration management
- `renderer.py`: Visualization renderer

---

### 3. HRG (Heterogeneous Resource Gathering) Environment

**Description**:  
An environment where heterogeneous agent teams collaborate to collect resources, with different roles having different capabilities and responsibilities.

**Key Features**:
- 🔍 **Heterogeneous Roles**: Scouts, workers, and transporters with different abilities
- 💰 **Resource Management**: Gold and wood resources with different values and collection difficulties
- 👁️ **Partial Observability**: Different roles have different vision ranges
- 🚧 **Obstacles**: Randomly distributed impassable areas
- 🏭 **Base System**: Resources must be returned to base for rewards
- 👥 **Agent Configuration**: 2 scouts + 3 workers + 1 transporter
- 📐 **Grid Size**: 10×10 grid (configurable)

**Key Parameters**:
- Action Space: 8 discrete actions (move, gather, transfer, deposit, wait)
- Observation Space: 80 dimensional vector (self state, visible entities, global info)
- Global State Dimension: 41 dimensions
- Max Steps: 200-300 steps (difficulty-dependent)
- Resources: 3 gold mines (value 10/unit) + 10 wood (value 2/unit)

**Files**:
- `env_hrg.py`: Base environment implementation
- `env_hrg_ctde.py`: CTDE-compatible version
- `core.py`: Core classes (Agent, Resource, etc.)
- `config.py`: Configuration management
- `renderer.py`: Visualization renderer

---

### 4. MSFS (Multi-agent Smart Factory Scheduling) Environment

**Description**:  
A smart manufacturing environment where robot agents collaborate to process orders, naturally forming roles through specialized reward signals.

**Key Features**:
- 🏭 **Manufacturing Flow**: Raw materials → Assembly → Packing (3 workstations)
- 📦 **Order System**: Standard and urgent orders
- 🤖 **Role Differentiation**: Agents specialize as collectors, assemblers, and packers
- ⚡ **Action Cooldowns**: Movement and processing have cooldown time limits
- 📊 **Utilization Tracking**: Tracks workstation and agent utilization
- 👥 **Agent Count**: 6 robots (configurable)
- 🏢 **Workstations**: 3 workstations (raw materials, assembly, packing)

**Key Parameters**:
- Action Space: 8 discrete actions (move to workstation, pick, place, process, etc.)
- Observation Space: 24 dimensional vector (self state, global info, teammate info)
- Global State Dimension: 42 dimensions
- Max Steps: 200-300 steps (difficulty-dependent)
- Order Generation: Dynamic generation of standard and urgent orders

**Files**:
- `env_msfs.py`: Base environment implementation
- `env_msfs_ctde.py`: CTDE-compatible version
- `core.py`: Core classes (Order, Workstation, Agent, etc.)
- `config.py`: Configuration management
- `renderer.py`: Visualization renderer

---

### 5. SMAC (StarCraft Multi-Agent Challenge) Wrapper

**Description**:  
A wrapper for the SMAC library providing a unified interface consistent with other environments, for StarCraft II multi-agent combat scenarios.

**Key Features**:
- ⚔️ **Real Combat**: Based on StarCraft II game engine
- 🗺️ **Multiple Maps**: Supports standard maps like 8m, 3s, 2s3z, MMM, corridor
- 🎯 **Heterogeneous Units**: Different unit types with different abilities
- 🔄 **Unified Interface**: Interface consistent with DEM/HRG/MSFS environments
- 🎮 **Action Masking**: Supports action availability detection
- 👥 **Agent Count**: 3-20 agents (map-dependent)
- 🌟 **Standard Benchmark**: Widely used MARL benchmark in the industry

**Key Parameters**:
- Action Space: ~14 discrete actions (varies by map)
- Observation Space: ~80 dimensional vector (varies by map)
- Max Steps: Preset by map
- Requires StarCraft II and SMAC library

**Files**:
- `env_smac.py`: SMAC wrapper implementation
- `env_smac_ctde.py`: CTDE-compatible version
- `config.py`: Configuration management
- `demo_wrapper.py`: Demo wrapper

---

## 🔧 Unified Interface Design

All environments follow this unified interface specification:

### Base Environment Interface

```python
# Reset environment
observations = env.reset()
# Returns: Dict[agent_id, np.ndarray]

# Execute actions
observations, rewards, dones, infos = env.step(actions)
# Arguments: actions: Dict[agent_id, int]
# Returns: observations, rewards, dones, infos (all in dictionary format)

# Get available actions
avail_actions = env.get_avail_actions(agent_id)

# Get environment info
env_info = env.get_env_info()

# Close environment
env.close()
```

### CTDE Environment Interface

CTDE (Centralized Training Decentralized Execution) environments add to the base interface:

```python
# Get global state
global_state = env.get_global_state()
# Returns: np.ndarray

# Global state included in info
obs, rewards, dones, infos = env.step(actions)
global_state = infos['global_state']

# Supported global state types
# - "concat": Concatenate all agent observations
# - "mean": Mean pooling
# - "max": Max pooling
# - "attention": Attention-based state representation
```

---

## 📦 Directory Structure

```
Env/
├── CM/                          # Collaborative Moving Environment
│   ├── env_cm.py               # Base environment
│   ├── env_cm_ctde.py          # CTDE version
│   ├── core.py                 # Core classes
│   ├── config.py               # Configuration
│   ├── renderer.py             # Renderer
│   ├── test_env.py             # Test file
│   └── README.md               # Detailed documentation
│
├── DEM/                         # Dynamic Escort Mission Environment
│   ├── env_dem.py              # Base environment
│   ├── env_dem_ctde.py         # CTDE version
│   ├── core.py                 # Core classes
│   ├── config.py               # Configuration
│   ├── renderer.py             # Renderer
│   ├── test_env.py             # Test file
│   └── README.md               # Detailed documentation
│
├── HRG/                         # Heterogeneous Resource Gathering Environment
│   ├── env_hrg.py              # Base environment
│   ├── env_hrg_ctde.py         # CTDE version
│   ├── core.py                 # Core classes
│   ├── config.py               # Configuration
│   ├── renderer.py             # Renderer
│   ├── test_env.py             # Test file
│   └── README.md               # Detailed documentation
│
├── MSFS/                        # Smart Factory Scheduling Environment
│   ├── env_msfs.py             # Base environment
│   ├── env_msfs_ctde.py        # CTDE version
│   ├── core.py                 # Core classes
│   ├── config.py               # Configuration
│   ├── renderer.py             # Renderer
│   ├── test_env.py             # Test file
│   └── README.md               # Documentation (TBD)
│
├── SMAC/                        # StarCraft Environment Wrapper
│   ├── env_smac.py             # SMAC wrapper
│   ├── env_smac_ctde.py        # CTDE version
│   ├── config.py               # Configuration
│   ├── test_env.py             # Test file
│   └── README.md               # Detailed documentation
│
├── doc/                         # Environment documentation and tutorials
│   ├── CM简介.md
│   ├── DEM简介.md
│   ├── HRG简介.md
│   ├── MSFS简介.md
│   └── requirements.txt
│
├── CM_Tutorial.ipynb           # CM environment tutorial
├── DEM_environment_tutorial.ipynb  # DEM environment tutorial
├── HRG_Tutorial.ipynb          # HRG environment tutorial
├── MSFS_environment_tutorial.ipynb # MSFS environment tutorial
├── SMAC_Wrapper_Tutorial.ipynb # SMAC wrapper tutorial
│
├── verify_dem_environment.py   # DEM environment verification script
├── verify_hrg_environment.py   # HRG environment verification script
├── verify_msfs_environment.py  # MSFS environment verification script
├── run_dem_validation.py       # Run DEM validation
├── run_hrg_validation.py       # Run HRG validation
├── run_msfs_validation.py      # Run MSFS validation
│
├── README_CN.md                # Chinese README
└── README_EN.md                # This file (English version)
```

---

## 🎓 Quick Start

### 1. CM Environment Example

```python
from Env.CM import create_cm_env

# Create environment
env = create_cm_env(difficulty="easy")

# Reset environment
observations = env.reset()

# Run an episode
for step in range(100):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 2. DEM Environment Example

```python
from Env.DEM import create_dem_env

# Create environment
env = create_dem_env(difficulty="normal")

# Reset environment
observations = env.reset()

# Run an episode
for step in range(200):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 3. HRG Environment Example

```python
from Env.HRG import create_hrg_env

# Create environment
env = create_hrg_env(difficulty="normal")

# Reset environment
observations = env.reset()

# Run an episode
for step in range(200):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 4. MSFS Environment Example

```python
from Env.MSFS import create_msfs_env

# Create environment
env = create_msfs_env(difficulty="normal")

# Reset environment
observations = env.reset()

# Run an episode
for step in range(200):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 5. CTDE Environment Example (for QMIX, etc.)

```python
from Env.CM import create_cm_ctde_env

# Create CTDE environment
env = create_cm_ctde_env(
    difficulty="normal_ctde",
    global_state_type="concat"
)

# Reset and get global state
observations = env.reset()
global_state = env.get_global_state()

# Run an episode
for step in range(100):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    # Get global state
    global_state = infos['global_state']
    
    if any(dones.values()):
        break

env.close()
```

---

## 🔬 Environment Comparison

| Environment | Agents | Cooperation | Obs Dim | Actions | Heterogeneous | Communication | Main Challenge |
|-------------|--------|-------------|---------|---------|---------------|---------------|----------------|
| **CM** | 2-4 | ⭐⭐ | 10-16 | 5 | No | No | Spatial coordination |
| **DEM** | 3 | ⭐⭐⭐⭐ | ~60 | 10 | No | Yes | Role emergence, dynamic threats |
| **HRG** | 6 | ⭐⭐⭐ | 80 | 8 | Yes | Optional | Heterogeneous cooperation, resource optimization |
| **MSFS** | 6 | ⭐⭐⭐ | 24 | 8 | No | No | Task allocation, temporal optimization |
| **SMAC** | 3-20 | ⭐⭐⭐⭐⭐ | ~80 | ~14 | Yes | No | Combat strategy, micro-control |

---

## 📊 Supported MARL Algorithms

All environments are compatible with mainstream MARL algorithms:

### CTDE Algorithms
- **QMIX**: Q-Mixing Networks
- **VDN**: Value Decomposition Networks
- **QTRAN**: Q-Transformation
- **WQMIX**: Weighted QMIX

### Independent Learning Algorithms
- **IQL**: Independent Q-Learning
- **A3C**: Asynchronous Advantage Actor-Critic
- **PPO**: Proximal Policy Optimization

### Communication Algorithms
- **CommNet**: Communication Networks
- **TarMAC**: Targeted Multi-Agent Communication
- **IC3Net**: Individual-Collective-Learning Communication

---

## 🧪 Testing and Validation

Each environment provides a complete test suite:

```bash
# CM environment tests
cd Env/CM
python test_env.py

# DEM environment tests
cd Env/DEM
python test_env.py

# HRG environment tests
cd Env/HRG
python test_env.py

# MSFS environment tests
cd Env/MSFS
python test_env.py

# SMAC environment tests
cd Env/SMAC
python test_env.py
```

Run complete validation scripts:

```bash
# Validate all environments
python verify_dem_environment.py
python verify_hrg_environment.py
python verify_msfs_environment.py
```

---

## 📚 Tutorials and Documentation

Each environment provides detailed Jupyter tutorials:

- **CM_Tutorial.ipynb**: Complete tutorial for Collaborative Moving environment
- **DEM_environment_tutorial.ipynb**: Dynamic Escort Mission environment tutorial
- **HRG_Tutorial.ipynb**: Heterogeneous Resource Gathering environment tutorial
- **MSFS_environment_tutorial.ipynb**: Smart Manufacturing environment tutorial
- **SMAC_Wrapper_Tutorial.ipynb**: SMAC wrapper usage tutorial

Detailed documentation is available in each environment's README.md file.

---

## ⚙️ Environment Configuration Guide

### Difficulty Levels

All environments support multiple predefined difficulty levels:

- **easy**: Suitable for initial training and debugging
- **normal**: Standard evaluation configuration
- **hard**: Challenging configuration for testing algorithm limits

### Custom Configuration

Each environment supports custom configuration:

```python
from Env.CM.config import CMConfig
from Env.CM.env_cm import CooperativeMovingEnv

# Create custom configuration
config = CMConfig(
    grid_size=9,
    n_agents=4,
    max_steps=120,
    cooperation_reward=0.03
)

# Use custom configuration
env = CooperativeMovingEnv(config)
```

---

## 🎨 Visualization Support

All environments provide visualization features:

### Text Rendering

```python
env = create_cm_env(render_mode="human")
env.reset()
env.render()  # Display text rendering in terminal
```

### Graphics Rendering

```python
env = create_dem_env(render_mode="rgb_array")
env.reset()

# Real-time visualization with Pygame
for step in range(100):
    actions = get_actions()
    env.step(actions)
    # Automatically renders
```

### Save Rendered Images

```python
from Env.HRG.renderer import MatplotlibRenderer

renderer = MatplotlibRenderer(grid_size=10)
renderer.render(env.game_state, save_path="screenshot.png")
```

---

## 🔄 Interface Consistency Guarantee

All environments strictly follow these interface specifications:

1. **reset() method**: Returns observation dictionary
2. **step() method**: Returns (observations, rewards, dones, infos) tuple
3. **Observation format**: Dict[agent_id, np.ndarray]
4. **Reward format**: Dict[agent_id, float]
5. **Done flags**: Dict[agent_id, bool]
6. **Info dictionary**: Dict containing global information

This ensures algorithm code can seamlessly switch between different environments.

---

## 🤝 Contribution Guidelines

Contributions to the environment collection are welcome!

### Adding New Environments

New environments should:
1. Follow the unified interface specification
2. Provide both base and CTDE versions
3. Include a complete configuration system
4. Provide test suites
5. Include detailed documentation and tutorials

### Code Standards

- Follow PEP 8 code style
- Add type annotations
- Write detailed docstrings
- Provide unit tests

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Thanks to the multi-agent reinforcement learning research community for their support and contributions.

---

## 📧 Contact

For questions or suggestions, please submit through GitHub Issues.

---

**Version**: v1.0.0  
**Last Updated**: 2025  
**Maintainers**: Shuwei Sun
