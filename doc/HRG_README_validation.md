# HRG环境验证程序

## 概述

本目录包含HRG（Heterogeneous Resource Gathering）环境的验证程序，用于验证环境实现与文档说明的一致性。

## 文件说明

- `verify_hrg_environment.py` - 主要验证程序
- `run_hrg_validation.py` - 运行脚本
- `validation_report.md` - 详细验证报告
- `README_validation.md` - 本说明文件

## 快速开始

### 基本验证

```bash
cd tutorials
python run_hrg_validation.py
```

这将运行normal难度的验证，检查核心功能是否与文档一致。

### 全面验证

```bash
cd tutorials
python run_hrg_validation.py --comprehensive
```

这将测试所有难度（easy, normal, hard）的配置。

## 验证内容

### 1. 观测空间验证（80维）

- ✅ 维度一致性检查
- ✅ 数据类型验证（float32）
- ✅ 数值范围合理性检查
- ✅ 结构组成分析：
  - 自身状态（11维）：位置、角色、库存、能量
  - 可见实体信息（50维）：最多10个实体
  - 通信历史（19维）：消息历史

### 2. 动作空间验证（8维）

- ✅ 动作维度检查
- ⚠️  角色特定动作限制验证
  - 侦察兵：不能采集、不能存放
  - 工人：不能存放
  - 运输车：不能采集（**发现问题**）
- ✅ 动作执行测试

### 3. 奖励机制验证

- ✅ 奖励数据类型和范围
- ✅ 多步采样统计分析
- ✅ 特定动作奖励验证：
  - 无效移动惩罚：-0.1
  - 时间惩罚：-0.01/步
  - 资源价值奖励机制

### 4. 配置验证

- ✅ 智能体能力配置
- ✅ 资源配置
- ✅ 环境参数

## 验证结果

### ✅ 通过的验证

1. **观测空间** - 完全符合文档
2. **智能体配置** - 角色能力正确
3. **资源配置** - 参数设置正确
4. **基本奖励机制** - 功能正常

### ❌ 发现的问题

1. **运输车采集动作限制**
   - 位置：`Env/HRG/core.py:202-203`
   - 问题：运输车可以执行GATHER动作
   - 影响：违反角色分工设计
   - 建议：添加运输车采集限制

### ⚠️ 注意事项

1. 通信历史实际占用19维，文档说10维（实现细节差异）

## 问题修复建议

### 运输车采集限制修复

在 `Env/HRG/core.py` 的 `can_perform_action` 方法中：

```python
# 当前代码（第202-203行）
if action == ActionType.GATHER and self.type == AgentType.SCOUT:
    return False

# 建议修改为
if action == ActionType.GATHER and self.type in [AgentType.SCOUT, AgentType.TRANSPORTER]:
    return False
```

## 扩展验证

可以扩展验证程序以检查：

1. **特定场景测试**
   - 资源采集完整流程
   - 智能体协作场景
   - 边界条件测试

2. **性能验证**
   - 环境执行速度
   - 内存使用效率
   - 大规模测试

3. **算法兼容性**
   - CTDE接口验证
   - 不同算法适配测试

## 集成到开发流程

建议将验证程序集成到开发流程中：

1. **代码提交前验证**
   ```bash
   python run_hrg_validation.py
   ```

2. **CI/CD集成**
   - 自动运行验证
   - 失败时阻止合并

3. **定期验证**
   - 每周运行全面验证
   - 跟踪环境变化

## 自定义验证

可以扩展 `HRGEnvironmentValidator` 类来添加自定义验证：

```python
class CustomValidator(HRGEnvironmentValidator):
    def validate_custom_feature(self):
        """自定义验证逻辑"""
        pass

    def run_validation(self, difficulty="normal"):
        # 运行标准验证
        super().run_validation(difficulty)

        # 运行自定义验证
        self.validate_custom_feature()
```

## 贡献指南

如果发现新的问题或需要添加验证项目：

1. 在相应验证方法中添加检查逻辑
2. 更新报告生成逻辑
3. 测试验证程序
4. 更新文档

## 技术细节

### 依赖项

- numpy
- sys, os, typing
- HRG环境模块

### 验证原理

1. **静态分析** - 检查配置和常量
2. **动态测试** - 执行环境并观察行为
3. **结构验证** - 验证数据结构和类型
4. **功能测试** - 测试特定功能

### 性能考虑

- 验证程序设计为轻量级
- 不会修改环境状态
- 可重复执行
- 适合频繁运行

---

**维护者**: Claude Code
**最后更新**: 2025-01-07
**版本**: 1.0