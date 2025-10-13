# 多智能体强化学习环境集合 (MARL Environments)

## 📖 简介

本文件夹包含了一套专为多智能体强化学习(MARL)设计的环境集合。所有环境都遵循统一的接口规范，支持集中式训练分布式执行(CTDE)算法，如QMIX、VDN、MADDPG等。

## 🎯 系统要求

- **操作系统**: Ubuntu 20.04+ (推荐 Ubuntu 22.04/24.04)
- **Python版本**: Python 3.8+
- **核心依赖**: NumPy, Gymnasium, Pygame, Matplotlib

## 🏗️ 环境列表

### 1. CM (Collaborative Moving) - 协作搬运环境

**环境描述**:  
一个简单而有效的多智能体协作环境，智能体需要合作将一个2×2的箱子推到指定的目标区域。

**核心特点**:
- 🤝 **协作机制**: 多个智能体需要从不同侧面推箱子，成功率随协作人数增加
- 🎮 **简单操作**: 5个离散动作（停留、上下左右移动）
- 🏆 **团队奖励**: 所有智能体获得相同的团队奖励
- ⚙️ **难度可配**: 支持easy、normal、hard三种难度级别
- 👥 **智能体数量**: 2-4个智能体（可配置）
- 📐 **网格大小**: 7×7网格（可配置）

**关键参数**:
- 动作空间: 5个离散动作
- 观察空间: 6 + 2×(n_agents-1) 维向量
- 最大步数: 50-100步（根据难度）

**文件**:
- `env_cm.py`: 基础环境实现
- `env_cm_ctde.py`: CTDE兼容版本
- `core.py`: 核心类定义
- `config.py`: 配置管理
- `renderer.py`: 可视化渲染

---

### 2. DEM (Dynamic Escort Mission) - 动态护送任务环境

**环境描述**:  
一个多智能体强化学习环境，智能体需要动态形成角色来护送VIP穿越危险区域，同时应对各种威胁。

**核心特点**:
- 🎭 **角色涌现**: 智能体自然形成防御者、护卫者、侦察者等角色
- 🗺️ **复杂地形**: 河流和森林影响移动和战斗
- 🎯 **动态威胁**: 敌人根据VIP状态自适应生成
- 💬 **通信机制**: 支持智能体之间的信息交流
- 🏥 **VIP保护**: 核心目标是保护VIP安全到达目的地
- 👥 **智能体数量**: 3个特种部队智能体（可配置）
- 📐 **网格大小**: 10×12 或 12×12（根据难度）

**关键参数**:
- 动作空间: 10个离散动作（移动、攻击、观察、守护、通信等）
- 观察空间: 约60维向量（包含自身状态、VIP状态、队友状态、威胁状态等）
- 全局状态维度: 41维
- 最大步数: 100-200步（根据难度）

**文件**:
- `env_dem.py`: 基础环境实现
- `env_dem_ctde.py`: CTDE兼容版本
- `core.py`: 核心类（Agent、VIP、Threat等）
- `config.py`: 配置管理
- `renderer.py`: 可视化渲染

---

### 3. HRG (Heterogeneous Resource Gathering) - 异构资源收集环境

**环境描述**:  
一个异构智能体团队协作收集资源的环境，不同角色的智能体具有不同的能力和职责。

**核心特点**:
- 🔍 **异构角色**: 侦察兵、工人、运输车三种不同角色
- 💰 **资源管理**: 金币和木材两种资源，价值和采集难度不同
- 👁️ **部分可观测**: 不同角色具有不同的视野范围
- 🚧 **障碍物**: 随机分布的不可通行区域
- 🏭 **基地系统**: 资源需要运回基地才能获得奖励
- 👥 **智能体配置**: 2个侦察兵 + 3个工人 + 1个运输车
- 📐 **网格大小**: 10×10网格（可配置）

**关键参数**:
- 动作空间: 8个离散动作（移动、采集、传输、存放、等待）
- 观察空间: 80维向量（自身状态、视野内实体、全局信息）
- 全局状态维度: 41维
- 最大步数: 200-300步（根据难度）
- 资源配置: 3个金矿（价值10/单位）+ 10个木材（价值2/单位）

**文件**:
- `env_hrg.py`: 基础环境实现
- `env_hrg_ctde.py`: CTDE兼容版本
- `core.py`: 核心类（Agent、Resource等）
- `config.py`: 配置管理
- `renderer.py`: 可视化渲染

---

### 4. MSFS (Multi-agent Smart Factory Scheduling) - 智能制造流程调度环境

**环境描述**:  
一个智能制造环境，机器人智能体需要协作处理订单，通过专业化奖励信号自然形成角色。

**核心特点**:
- 🏭 **制造流程**: 原材料 → 组装 → 包装三个工作站
- 📦 **订单系统**: 标准订单和紧急订单两种类型
- 🤖 **角色分化**: 智能体通过专业化形成采集者、组装者、包装者角色
- ⚡ **动作冷却**: 移动和处理都有冷却时间限制
- 📊 **利用率追踪**: 跟踪工作站和智能体的利用率
- 👥 **智能体数量**: 6个机器人（可配置）
- 🏢 **工作站**: 3个工作站（原材料、组装、包装）

**关键参数**:
- 动作空间: 8个离散动作（移动到工作站、拾取、放置、处理等）
- 观察空间: 24维向量（自身状态、全局信息、队友信息）
- 全局状态维度: 42维
- 最大步数: 200-300步（根据难度）
- 订单生成: 动态生成，标准订单和紧急订单

**文件**:
- `env_msfs.py`: 基础环境实现
- `env_msfs_ctde.py`: CTDE兼容版本
- `core.py`: 核心类（Order、Workstation、Agent等）
- `config.py`: 配置管理
- `renderer.py`: 可视化渲染

---

### 5. SMAC (StarCraft Multi-Agent Challenge) - 星际争霸多智能体挑战环境封装

**环境描述**:  
基于SMAC库的封装器，提供与其他环境统一的接口，用于星际争霸II的多智能体对战场景。

**核心特点**:
- ⚔️ **真实战斗**: 基于星际争霸II游戏引擎
- 🗺️ **多种地图**: 支持8m、3s、2s3z、MMM、corridor等标准地图
- 🎯 **异构单位**: 不同单位类型具有不同能力
- 🔄 **统一接口**: 与DEM/HRG/MSFS环境接口一致
- 🎮 **动作掩码**: 支持动作可用性检测
- 👥 **智能体数量**: 3-20个（根据地图）
- 🌟 **标准基准**: 业界广泛使用的MARL基准环境

**关键参数**:
- 动作空间: 约14个离散动作（根据地图变化）
- 观察空间: 约80维向量（根据地图变化）
- 最大步数: 根据地图预设
- 需要StarCraft II和SMAC库

**文件**:
- `env_smac.py`: SMAC封装器实现
- `env_smac_ctde.py`: CTDE兼容版本
- `config.py`: 配置管理
- `demo_wrapper.py`: 演示包装器

---

## 🔧 统一接口设计

所有环境都遵循以下统一接口规范：

### 基础环境接口

```python
# 重置环境
observations = env.reset()
# 返回: Dict[agent_id, np.ndarray]

# 执行动作
observations, rewards, dones, infos = env.step(actions)
# 参数: actions: Dict[agent_id, int]
# 返回: observations, rewards, dones, infos (所有都是字典格式)

# 获取可用动作
avail_actions = env.get_avail_actions(agent_id)

# 获取环境信息
env_info = env.get_env_info()

# 关闭环境
env.close()
```

### CTDE环境接口

CTDE（集中式训练分布式执行）环境在基础接口之上增加了：

```python
# 获取全局状态
global_state = env.get_global_state()
# 返回: np.ndarray

# 在info中包含全局状态
obs, rewards, dones, infos = env.step(actions)
global_state = infos['global_state']

# 支持的全局状态类型
# - "concat": 拼接所有智能体观察
# - "mean": 平均池化
# - "max": 最大池化
# - "attention": 基于注意力机制的状态
```

---

## 📦 目录结构

```
Env/
├── CM/                          # 协作搬运环境
│   ├── env_cm.py               # 基础环境
│   ├── env_cm_ctde.py          # CTDE版本
│   ├── core.py                 # 核心类
│   ├── config.py               # 配置
│   ├── renderer.py             # 渲染器
│   ├── test_env.py             # 测试文件
│   └── README.md               # 详细文档
│
├── DEM/                         # 动态护送任务环境
│   ├── env_dem.py              # 基础环境
│   ├── env_dem_ctde.py         # CTDE版本
│   ├── core.py                 # 核心类
│   ├── config.py               # 配置
│   ├── renderer.py             # 渲染器
│   ├── test_env.py             # 测试文件
│   └── README.md               # 详细文档
│
├── HRG/                         # 异构资源收集环境
│   ├── env_hrg.py              # 基础环境
│   ├── env_hrg_ctde.py         # CTDE版本
│   ├── core.py                 # 核心类
│   ├── config.py               # 配置
│   ├── renderer.py             # 渲染器
│   ├── test_env.py             # 测试文件
│   └── README.md               # 详细文档
│
├── MSFS/                        # 智能制造调度环境
│   ├── env_msfs.py             # 基础环境
│   ├── env_msfs_ctde.py        # CTDE版本
│   ├── core.py                 # 核心类
│   ├── config.py               # 配置
│   ├── renderer.py             # 渲染器
│   ├── test_env.py             # 测试文件
│   └── README.md               # 详细文档（待补充）
│
├── SMAC/                        # 星际争霸环境封装
│   ├── env_smac.py             # SMAC封装器
│   ├── env_smac_ctde.py        # CTDE版本
│   ├── config.py               # 配置
│   ├── test_env.py             # 测试文件
│   └── README.md               # 详细文档
│
├── doc/                         # 环境文档和教程
│   ├── CM简介.md
│   ├── DEM简介.md
│   ├── HRG简介.md
│   ├── MSFS简介.md
│   └── requirements.txt
│
├── CM_Tutorial.ipynb           # CM环境教程
├── DEM_environment_tutorial.ipynb  # DEM环境教程
├── HRG_Tutorial.ipynb          # HRG环境教程
├── MSFS_environment_tutorial.ipynb # MSFS环境教程
├── SMAC_Wrapper_Tutorial.ipynb # SMAC封装器教程
│
├── verify_dem_environment.py   # DEM环境验证脚本
├── verify_hrg_environment.py   # HRG环境验证脚本
├── verify_msfs_environment.py  # MSFS环境验证脚本
├── run_dem_validation.py       # DEM验证运行
├── run_hrg_validation.py       # HRG验证运行
├── run_msfs_validation.py      # MSFS验证运行
│
└── README_CN.md                # 本文件（中文版）
```

---

## 🎓 快速开始

### 1. CM环境示例

```python
from Env.CM import create_cm_env

# 创建环境
env = create_cm_env(difficulty="easy")

# 重置环境
observations = env.reset()

# 运行一个回合
for step in range(100):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 2. DEM环境示例

```python
from Env.DEM import create_dem_env

# 创建环境
env = create_dem_env(difficulty="normal")

# 重置环境
observations = env.reset()

# 运行一个回合
for step in range(200):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 3. HRG环境示例

```python
from Env.HRG import create_hrg_env

# 创建环境
env = create_hrg_env(difficulty="normal")

# 重置环境
observations = env.reset()

# 运行一个回合
for step in range(200):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 4. MSFS环境示例

```python
from Env.MSFS import create_msfs_env

# 创建环境
env = create_msfs_env(difficulty="normal")

# 重置环境
observations = env.reset()

# 运行一个回合
for step in range(200):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    if any(dones.values()):
        break

env.close()
```

### 5. CTDE环境示例（适用于QMIX等算法）

```python
from Env.CM import create_cm_ctde_env

# 创建CTDE环境
env = create_cm_ctde_env(
    difficulty="normal_ctde",
    global_state_type="concat"
)

# 重置环境并获取全局状态
observations = env.reset()
global_state = env.get_global_state()

# 运行一个回合
for step in range(100):
    actions = {agent_id: env.action_space.sample() 
               for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)
    
    # 获取全局状态
    global_state = infos['global_state']
    
    if any(dones.values()):
        break

env.close()
```

---

## 🔬 环境对比

| 环境 | 智能体数 | 协作难度 | 观察维度 | 动作数 | 是否异构 | 通信支持 | 主要挑战 |
|------|---------|---------|---------|--------|---------|---------|---------|
| **CM** | 2-4 | ⭐⭐ | 10-16 | 5 | 否 | 否 | 空间协调 |
| **DEM** | 3 | ⭐⭐⭐⭐ | ~60 | 10 | 否 | 是 | 角色涌现、动态威胁 |
| **HRG** | 6 | ⭐⭐⭐ | 80 | 8 | 是 | 可选 | 异构协作、资源优化 |
| **MSFS** | 6 | ⭐⭐⭐ | 24 | 8 | 否 | 否 | 任务分配、时序优化 |
| **SMAC** | 3-20 | ⭐⭐⭐⭐⭐ | ~80 | ~14 | 是 | 否 | 战斗策略、微操控制 |

---

## 📊 支持的MARL算法

所有环境都兼容以下主流MARL算法：

### CTDE类算法
- **QMIX**: Q-Mixing Networks
- **VDN**: Value Decomposition Networks
- **QTRAN**: Q-Transformation
- **WQMIX**: Weighted QMIX

### 独立学习算法
- **IQL**: Independent Q-Learning
- **A3C**: Asynchronous Advantage Actor-Critic
- **PPO**: Proximal Policy Optimization

### 通信算法
- **CommNet**: Communication Networks
- **TarMAC**: Targeted Multi-Agent Communication
- **IC3Net**: Individual-Collective-Learning Communication

---

## 🧪 测试和验证

每个环境都提供了完整的测试套件：

```bash
# CM环境测试
cd Env/CM
python test_env.py

# DEM环境测试
cd Env/DEM
python test_env.py

# HRG环境测试
cd Env/HRG
python test_env.py

# MSFS环境测试
cd Env/MSFS
python test_env.py

# SMAC环境测试
cd Env/SMAC
python test_env.py
```

运行完整验证脚本：

```bash
# 验证所有环境
python verify_dem_environment.py
python verify_hrg_environment.py
python verify_msfs_environment.py
```

---

## 📚 教程和文档

每个环境都提供了详细的Jupyter教程：

- **CM_Tutorial.ipynb**: 协作搬运环境完整教程
- **DEM_environment_tutorial.ipynb**: 动态护送任务环境教程
- **HRG_Tutorial.ipynb**: 异构资源收集环境教程
- **MSFS_environment_tutorial.ipynb**: 智能制造环境教程
- **SMAC_Wrapper_Tutorial.ipynb**: SMAC封装器使用教程

详细文档位于各环境的README.md文件中。

---

## ⚙️ 环境配置指南

### 难度级别

所有环境都支持多个预定义难度级别：

- **easy**: 适合初始训练和调试
- **normal**: 标准评估配置
- **hard**: 挑战性配置，用于测试算法极限

### 自定义配置

每个环境都支持自定义配置：

```python
from Env.CM.config import CMConfig
from Env.CM.env_cm import CooperativeMovingEnv

# 创建自定义配置
config = CMConfig(
    grid_size=9,
    n_agents=4,
    max_steps=120,
    cooperation_reward=0.03
)

# 使用自定义配置
env = CooperativeMovingEnv(config)
```

---

## 🎨 可视化支持

所有环境都提供了可视化功能：

### 文本渲染

```python
env = create_cm_env(render_mode="human")
env.reset()
env.render()  # 在终端显示文本渲染
```

### 图形渲染

```python
env = create_dem_env(render_mode="rgb_array")
env.reset()

# 使用Pygame实时可视化
for step in range(100):
    actions = get_actions()
    env.step(actions)
    # 自动渲染
```

### 保存渲染图像

```python
from Env.HRG.renderer import MatplotlibRenderer

renderer = MatplotlibRenderer(grid_size=10)
renderer.render(env.game_state, save_path="screenshot.png")
```

---

## 🔄 接口一致性保证

所有环境严格遵循以下接口规范：

1. **reset()方法**: 返回观察字典
2. **step()方法**: 返回(observations, rewards, dones, infos)四元组
3. **观察格式**: Dict[agent_id, np.ndarray]
4. **奖励格式**: Dict[agent_id, float]
5. **完成标志**: Dict[agent_id, bool]
6. **信息字典**: Dict包含全局信息

这确保了算法代码可以在不同环境间无缝切换。

---

## 🤝 贡献指南

欢迎为环境集合做出贡献！

### 添加新环境

新环境应该：
1. 遵循统一接口规范
2. 提供基础版本和CTDE版本
3. 包含完整的配置系统
4. 提供测试套件
5. 编写详细文档和教程

### 代码规范

- 遵循PEP 8代码风格
- 添加类型注解
- 编写详细的文档字符串
- 提供单元测试

---

## 📄 许可证

本项目采用MIT许可证。

---

## 🙏 致谢

感谢多智能体强化学习研究社区的支持和贡献。

---

## 📧 联系方式

如有问题或建议，请通过GitHub Issues提交。

---

**版本**: v1.0.0  
**最后更新**: 2025年  
**维护者**: Shuwei Sun
