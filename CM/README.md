# CM (Collaborative Moving) Environment

一个简单而有效的多智能体协作搬运环境，专为测试多智能体强化学习算法的收敛性而设计。

## 📋 目录

- [环境简介](#环境简介)
- [核心特性](#核心特性)
- [安装和使用](#安装和使用)
- [环境规则](#环境规则)
- [配置选项](#配置选项)
- [API文档](#api文档)
- [示例代码](#示例代码)
- [教程](#教程)
- [测试](#测试)

## 🎯 环境简介

CM环境是一个多智能体协作任务，智能体需要合作将一个2x2的箱子推到指定的2x2目标区域。环境设计简洁但具有挑战性，特别适合验证MARL算法的基础能力。

### 游戏场景
- **网格世界**：7x7网格（可配置）
- **箱子**：占据2x2格子
- **目标**：2x2的目标区域
- **智能体**：2-4个智能体（可配置）

## ✨ 核心特性

- 🤝 **协作机制**：多个智能体从不同侧面推箱子，成功率随协作人数增加
- 🎮 **简单操作**：5个离散动作（停留、上下左右移动）
- 🏆 **团队奖励**：所有智能体获得相同的团队奖励
- ⚙️ **高度可配置**：支持多种难度级别和自定义配置
- 🔗 **CTDE兼容**：完美支持QMIX、VDN、IQL等主流MARL算法
- 📊 **丰富观测**：包含自身位置、箱子位置、目标位置和其他智能体相对位置
- 🎨 **可视化支持**：提供文本和图形渲染功能

## 📥 安装和使用

### 基础使用

```python
from Env.CM import create_cm_env

# 创建环境
env = create_cm_env(difficulty="easy", render_mode="human")

# 重置环境
observations, info = env.reset()

# 执行动作
actions = {agent_id: env.get_avail_actions(agent_id)[0] for agent_id in env.agent_ids}
observations, rewards, terminated, truncated, info = env.step(actions)

# 渲染环境
env.render()

# 关闭环境
env.close()
```

### CTDE环境（用于MARL算法）

```python
from Env.CM import create_cm_ctde_env

# 创建CTDE环境
ctde_env = create_cm_ctde_env(
    difficulty="easy_ctde",
    global_state_type="concat"
)

# 重置环境（返回全局状态）
observations, global_state = ctde_env.reset()

# 执行动作
actions = {agent_id: 0 for agent_id in ctde_env.agent_ids}
observations, rewards, terminated, truncated, info = ctde_env.step(actions)

# 获取全局状态
current_global_state = info['global_state']

ctde_env.close()
```

## 🎮 环境规则

### 协作推动机制
- **1个智能体推动**：50%成功率
- **2个智能体协作**：75%成功率
- **3个智能体协作**：90%成功率
- **4个智能体协作**：100%成功率

### 奖励机制
- **时间惩罚**：-0.01/步（鼓励快速完成）
- **碰撞惩罚**：-0.1（智能体之间碰撞）
- **协作奖励**：+0.02 × 协作人数
- **距离奖励**：基于箱子与目标距离的变化
- **目标达成奖励**：+10.0

### 动作空间
- 0: STAY（停留）
- 1: MOVE_UP（向上移动）
- 2: MOVE_DOWN（向下移动）
- 3: MOVE_LEFT（向左移动）
- 4: MOVE_RIGHT（向右移动）

### 观测空间
每个智能体的观测包含：
- 自身位置 (x, y)
- 箱子中心位置 (x, y)
- 目标中心位置 (x, y)
- 其他智能体的相对位置（每个智能体2维）

## ⚙️ 配置选项

### 预定义难度

```python
from Env.CM import get_config_by_name

# 可用配置
configs = ["easy", "normal", "hard", "debug", "cooperation_test",
           "single_agent", "multi_agent", "easy_ctde", "normal_ctde", "hard_ctde"]

# 使用配置
config = get_config_by_name("easy")
env = create_cm_env_from_config(config)
```

### 自定义配置

```python
from Env.CM import CMConfig, create_cm_env_from_config

# 创建自定义配置
custom_config = CMConfig(
    grid_size=9,           # 网格大小
    n_agents=3,            # 智能体数量
    max_steps=120,         # 最大步数
    push_success_probs={   # 推动成功率
        1: 0.6, 2: 0.85, 3: 1.0, 4: 1.0
    },
    cooperation_reward=0.03,       # 协作奖励
    goal_reached_reward=15.0      # 目标奖励
)

# 使用自定义配置
env = create_cm_env_from_config(custom_config)
```

## 📚 API文档

### 主要函数

#### `create_cm_env(difficulty="normal", **kwargs)`
创建基础CM环境。

**参数：**
- `difficulty`：预定义难度级别
- `**kwargs`：配置覆盖参数

**返回：** CooperativeMovingEnv实例

#### `create_cm_ctde_env(difficulty="normal_ctde", global_state_type="concat", **kwargs)`
创建CTDE兼容环境。

**参数：**
- `difficulty`：CTDE优化的难度级别
- `global_state_type`：全局状态类型（"concat", "mean", "max", "attention"）
- `**kwargs`：配置覆盖参数

**返回：** CooperativeMovingCTDEEnv实例

### 环境方法

#### `reset(seed=None)`
重置环境。

**返回：**
- `observations`：各智能体的观测
- `info`：环境信息

#### `step(actions)`
执行一步。

**参数：**
- `actions`：智能体动作字典

**返回：**
- `observations`：新观测
- `rewards`：奖励
- `terminated`：是否终止
- `truncated`：是否截断
- `info`：环境信息

#### `get_avail_actions(agent_id)`
获取智能体的可用动作。

**参数：**
- `agent_id`：智能体ID

**返回：** 可用动作列表

#### `render()`
渲染环境（如果启用）。

#### `get_env_info()`
获取环境信息。

**返回：** 包含环境参数的字典

## 💡 示例代码

### 完整回合示例

```python
from Env.CM import create_cm_env
import numpy as np

def run_random_episode():
    env = create_cm_env(difficulty="easy")
    obs, info = env.reset()

    total_reward = 0
    steps = 0

    while steps < 100:
        # 随机选择动作
        actions = {}
        for agent_id in env.agent_ids:
            avail_actions = env.get_avail_actions(agent_id)
            actions[agent_id] = np.random.choice(avail_actions)

        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions)

        total_reward += list(rewards.values())[0]
        steps += 1

        print(f"Step {steps}: reward={list(rewards.values())[0]:.3f}, "
              f"distance={info['distance_to_goal']:.2f}")

        if terminated:
            print(f"Goal reached in {steps} steps!")
            break
        elif truncated:
            print("Episode truncated!")
            break

    env.close()
    return total_reward, steps

# 运行示例
reward, steps = run_random_episode()
print(f"Episode result: reward={reward:.3f}, steps={steps}")
```

### 简单协作智能体

```python
from Env.CM import create_cm_env

class SimpleCooperativeAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def choose_action(self, env, observations):
        obs = observations[self.agent_id]
        agent_x, agent_y = obs[0], obs[1]
        box_x, box_y = obs[2], obs[3]
        goal_x, goal_y = obs[4], obs[5]

        # 简单策略：向箱子推动方向移动
        dx = goal_x - box_x
        dy = goal_y - box_y

        avail_actions = env.get_avail_actions(self.agent_id)

        if abs(dx) > abs(dy):  # 主要在x方向推动
            if dx > 0 and 2 in avail_actions:
                return 2  # MOVE_DOWN
            elif dx < 0 and 1 in avail_actions:
                return 1  # MOVE_UP
        else:  # 主要在y方向推动
            if dy > 0 and 4 in avail_actions:
                return 4  # MOVE_RIGHT
            elif dy < 0 and 3 in avail_actions:
                return 3  # MOVE_LEFT

        # 随机移动
        move_actions = [a for a in [1, 2, 3, 4] if a in avail_actions]
        return np.random.choice(move_actions) if move_actions else 0

# 使用协作智能体
env = create_cm_env(difficulty="easy")
agents = [SimpleCooperativeAgent(agent_id) for agent_id in env.agent_ids]

obs, info = env.reset()
for step in range(50):
    actions = {agent.agent_id: agent.choose_action(env, obs) for agent in agents}
    obs, rewards, terminated, truncated, info = env.step(actions)

    if terminated:
        print(f"Cooperative agents reached goal in {step} steps!")
        break

env.close()
```

## 📖 教程

详细教程请参考：
- [CM_Tutorial.ipynb](./CM_Tutorial.ipynb) - 完整的Jupyter教程
- [协作搬运游戏设计.md](../协作搬运游戏设计.md) - 设计文档

## 🧪 测试

运行测试套件：

```bash
# 基础功能测试
python test_env.py quick

# 完整测试套件
python test_env.py comprehensive

# 配置系统测试
python test_env.py config

# 最小化测试
python minimal_test.py
```

## 📊 性能指标

- **执行速度**：>1000 步/秒（debug配置）
- **内存使用**：<100MB（基础环境）
- **支持算法**：QMIX, VDN, IQL, MADDPG, MAPPO等
- **可扩展性**：支持1-4个智能体

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

本项目遵循MIT许可证。

## 🔗 相关项目

- [DEM环境](../DEM/) - 防御、护送和移动环境
- [HRG环境](../HRG/) - 异构资源收集环境
- [MSFS环境](../MSFS/) - 多智能体搜索救援环境