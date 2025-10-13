# HRG (Heterogeneous Resource Gathering) Environment

一个专为多智能体强化学习设计的异构资源收集环境，用于测试和训练具有不同角色的智能体团队的协作能力。

## 🌟 特性

- **异构智能体角色**：侦察兵(Scout)、工人(Worker)、运输车(Transporter)
- **部分可观测环境**：不同角色具有不同的视野范围和能力
- **资源管理系统**：金币(Gold)和木材(Wood)两种资源，具有不同价值和采集难度
- **CTDE兼容**：支持集中式训练分布式执行算法
- **可视化支持**：实时可视化智能体行为和环境状态
- **高度可配置**：支持多种难度设置和实验配置
- **稳健设计**：防止异常值，适合强化学习训练

## 📋 环境描述

### 世界设定
- **网格世界**：10×10的方形网格（可配置）
- **基地位置**：固定在(0, 0)
- **障碍物**：随机分布的不可通行区域
- **视野机制**：基于射线投射的有限视野

### 智能体角色

#### 🔍 侦察兵 (Scout) - 2个
- **视野范围**：5格
- **移动速度**：2格/步
- **负重能力**：0（无法采集资源）
- **特殊能力**：快速探索，发现远程资源
- **能量消耗**：0.05/步

#### ⚒️ 工人 (Worker) - 3个
- **视野范围**：3格
- **移动速度**：1格/步
- **负重能力**：2单位
- **采集时间**：2步/单位资源
- **能量消耗**：0.02/步，0.08/采集动作

#### 🚚 运输车 (Transporter) - 1个
- **视野范围**：4格
- **移动速度**：1.5格/步
- **负重能力**：5单位
- **传输时间**：1步（与工人相邻时）
- **能量消耗**：0.03/步，0.1/传输动作

### 资源配置

#### 💰 金矿 (Gold)
- **数量**：3个
- **分布**：聚集在地图远端(7-9, 7-9)区域
- **价值**：10分/单位
- **采集难度**：4步/单位（每个矿点2单位）

#### 🪵 木材 (Wood)
- **数量**：10个
- **分布**：均匀散布在地图(2-8, 2-8)区域
- **价值**：2分/单位
- **采集难度**：2步/单位（每个矿点1单位）

## 🎮 动作空间

每个智能体有8个离散动作：

1. **上移动** (MOVE_NORTH)
2. **下移动** (MOVE_SOUTH)
3. **左移动** (MOVE_WEST)
4. **右移动** (MOVE_EAST)
5. **采集资源** (GATHER) - 仅工人可用
6. **传输资源** (TRANSFER) - 工人与运输车之间
7. **存放资源** (DEPOSIT) - 仅运输车在基地可用
8. **等待** (WAIT)

## 👁️ 观察空间

每个智能体的观察向量为80维，包含：

### 自身状态 (10维)
- 位置：(x, y) 归一化坐标
- 角色ID：one-hot编码 (3维)
- 携带资源：(gold_count, wood_count)
- 能量：energy/100.0
- 动作冷却：cooldown/2.0

### 视野内实体 (50维，最多10个实体)
每个实体包含5维信息：
- 相对位置：(Δx, Δy)
- 实体类型：one-hot编码 (5维)
- 资源剩余量：count/5.0（仅资源点）

### 全局信息 (20维)
- 最近3条消息的编码
- 基地距离：归一化值
- 剩余时间：time_left/max_time
- 其他游戏状态信息

## 🏆 奖励机制

### 主要奖励
- **存放资源**：资源价值的50%
  - 金币：10分 × 50% = 5分
  - 木材：2分 × 50% = 1分

### 塑造奖励
- **采集资源**：资源价值的10%
- **传输资源**：资源价值的5%
- **发现新资源**：0.5分
- **成功传输**：0.2分

### 惩罚项
- **时间惩罚**：-0.01/步
- **无效移动**：-0.1
- **碰撞惩罚**：-0.05
- **通信成本**：-0.01/次（如果启用通信）

## 🚀 快速开始

### 基础使用

```python
from Env.HRG import create_hrg_env

# 创建环境
env = create_hrg_env(difficulty="normal")

# 重置环境
observations = env.reset()

# 执行动作
actions = {agent_id: env.action_space.sample() for agent_id in env.agent_ids}
observations, rewards, dones, infos = env.step(actions)

# 关闭环境
env.close()
```

### CTDE环境使用

```python
from Env.HRG import create_hrg_ctde_env

# 创建CTDE兼容环境
env = create_hrg_ctde_env(
    config_name="normal_ctde",
    global_state_type="concat"
)

# 获取环境信息
env_info = env.get_env_info()
print(f"Number of agents: {env_info['n_agents']}")
print(f"Global state dimension: {env_info['global_state_dim']}")

# 获取全局状态
global_state = env.get_global_state()
```

### 自定义配置

```python
from Env.HRG.env_hrg import HRGConfig, HRGEnv

# 创建自定义配置
config = HRGConfig(
    grid_size=12,
    max_steps=300,
    num_gold=5,
    num_wood=15,
    render_mode="human"
)

# 创建环境
env = HRGEnv(config)
```

## 🎯 实验配置

### 难度预设

```python
# 简单模式 - 适合初始训练
env = create_hrg_env(difficulty="easy")

# 标准模式 - 默认评估配置
env = create_hrg_env(difficulty="normal")

# 困难模式 - 挑战测试
env = create_hrg_env(difficulty="hard")
```

### 专用配置

```python
from Env.HRG.config import get_config_by_name

# 通信测试环境
config = get_config_by_name("communication")
env = HRGEnv(config)

# 协调测试环境
config = get_config_by_name("coordination")
env = HRGEnv(config)

# 探索测试环境
config = get_config_by_name("exploration")
env = HRGEnv(config)
```

## 📊 可视化

### Pygame实时可视化

```python
from Env.HRG.env_hrg import HRGConfig
from Env.HRG.renderer import HRGRenderer

config = HRGConfig(render_mode="human")
env = HRGEnv(config)

observations = env.reset()
for step in range(100):
    actions = {agent_id: np.random.randint(0, 8) for agent_id in env.agent_ids}
    obs, rewards, dones, infos = env.step(actions)

    if any(dones.values()):
        break

env.close()
```

### Matplotlib静态可视化

```python
from Env.HRG.renderer import MatplotlibRenderer

env = create_hrg_env(difficulty="normal")
env.reset()

renderer = MatplotlibRenderer(grid_size=env.config.grid_size)
renderer.render(env.game_state, save_path="hrg_snapshot.png")
```

## 🧪 测试和验证

### 运行完整测试套件

```python
from Env.HRG.test_env import run_comprehensive_test

# 运行全面测试
test_results = run_comprehensive_test()
```

### 快速测试

```python
from Env.HRG.test_env import run_quick_test

# 运行快速功能测试
run_quick_test()
```

## 📈 性能指标

环境提供以下评估指标：

```python
# 在info字典中获取
info = env.get_info(agent_id)

# 主要指标
'total_score': 总得分
'gold_deposited': 存放的金币数量
'wood_deposited': 存放的木材数量
'step': 当前步数
```

### 成功指标
- **总得分**：主要评估指标
- **金币收集率**：高价值资源收集效率
- **木材收集率**：基础资源收集效率
- **探索覆盖率**：地图探索程度
- **通信效率**：得分与通信次数的比值
- **角色利用率**：各角色的使用效率

## 🔧 高级功能

### 动作掩码

```python
# 获取可用动作（用于约束动作空间）
for agent_id in env.agent_ids:
    avail_actions = env.get_avail_actions(agent_id)
    # 仅选择可用动作
    action = np.random.choice(avail_actions)
    actions[agent_id] = action
```

### 环境信息获取

```python
# 获取详细环境信息
env_info = {
    'n_agents': env.n_agents,
    'agent_ids': env.agent_ids,
    'obs_dims': env.obs_dims,
    'act_dims': env.act_dims,
    'episode_limit': env.config.max_steps
}
```

### 全局状态类型

```python
# 不同类型的全局状态表示
env_types = [
    create_hrg_ctde_env("normal", global_state_type="concat"),  # 拼接
    create_hrg_ctde_env("normal", global_state_type="mean"),    # 平均
    create_hrg_ctde_env("normal", global_state_type="max"),     # 最大
    create_hrg_ctde_env("normal", global_state_type="attention") # 注意力
]
```

## 🎓 使用场景

### 1. 基础多智能体训练
- 测试算法在异构智能体环境中的表现
- 验证角色特化的有效性
- 评估协作策略学习

### 2. 通信协议学习
- 测试显式通信机制
- 评估隐式协调行为
- 研究通信成本与效率的权衡

### 3. 探索-利用权衡
- 评估不同探索策略
- 测试长期规划能力
- 验证资源分配优化

### 4. 课程学习
- 从简单到复杂的渐进式训练
- 技能学习和迁移
- 难度自适应调整

## ⚠️ 注意事项

1. **内存管理**：长时间训练时注意及时关闭环境
2. **随机种子**：设置种子以保证实验可重现性
3. **可视化性能**：实时可视化会影响训练速度
4. **动作约束**：建议使用动作掩码避免无效动作
5. **观察维度**：确保观察空间与算法要求匹配

## 🤝 贡献指南

欢迎提交问题报告和改进建议！

### 开发环境设置

```bash
# 安装依赖
pip install numpy pygame matplotlib gymnasium

# 运行测试
python Env/HRG/test_env.py

# 代码风格
遵循PEP 8规范，添加类型注解和文档字符串
```

## 📄 许可证

本项目采用MIT许可证。

## 🙏 致谢

感谢多智能体强化学习研究社区的启发和支持。

---

**HRG Environment v1.0.0**
专为多智能体强化学习研究设计的异构资源收集环境