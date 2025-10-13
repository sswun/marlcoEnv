#!/usr/bin/env python3
"""
MSFS环境验证程序运行脚本

使用方法:
    python run_msfs_validation.py
    python run_msfs_validation.py --comprehensive
    python run_msfs_validation.py --basic-only

作者: Claude Code
日期: 2025-01-08
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verify_msfs_environment import MSFSEnvironmentValidator


def run_quick_validation():
    """运行快速验证"""
    print("🚀 开始MSFS环境快速验证...")

    validator = MSFSEnvironmentValidator()

    try:
        # 只验证normal难度的CTDE模式，这是最常用的配置
        success = validator.run_validation("normal", True)

        if success:
            print("\n🎉 验证完成！MSFS环境实现与文档完全一致。")
            return 0
        else:
            print("\n⚠️  验证发现问题，请查看详细报告。")
            return 1

    except Exception as e:
        print(f"\n💥 验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def run_comprehensive_validation():
    """运行全面验证（测试所有难度和模式）"""
    print("🔬 开始MSFS环境全面验证...")

    test_configs = [
        ("normal", True, "CTDE模式"),
        ("normal", False, "基础模式"),
        ("easy", True, "简单难度CTDE"),
        ("hard", True, "困难难度CTDE"),
    ]

    validator = MSFSEnvironmentValidator()
    total_success = True

    try:
        for difficulty, use_ctde, description in test_configs:
            print(f"\n{'='*60}")
            print(f"验证配置: {description}")
            print('='*60)

            success = validator.run_validation(difficulty, use_ctde)

            if not success:
                total_success = False
                print(f"❌ {description} 验证失败")
            else:
                print(f"✅ {description} 验证通过")

            validator.cleanup()

            # 重新创建环境用于下一个测试
            validator = MSFSEnvironmentValidator()

        print(f"\n{'='*60}")
        if total_success:
            print("🎉 所有配置验证均通过！")
            return 0
        else:
            print("⚠️  部分配置验证失败")
            return 1

    except Exception as e:
        print(f"\n💥 验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def run_basic_validation():
    """只运行基础模式验证（不包含CTDE）"""
    print("🔧 开始MSFS环境基础模式验证...")

    validator = MSFSEnvironmentValidator()

    try:
        # 只验证基础模式
        success = validator.run_validation("normal", False)

        if success:
            print("\n🎉 基础模式验证完成！")
            return 0
        else:
            print("\n⚠️  基础模式验证发现问题。")
            return 1

    except Exception as e:
        print(f"\n💥 基础模式验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def run_ctde_validation():
    """只运行CTDE模式验证"""
    print("🧠 开始MSFS环境CTDE模式验证...")

    validator = MSFSEnvironmentValidator()

    try:
        # 验证CTDE模式的所有难度
        difficulties = ["easy", "normal", "hard"]
        all_success = True

        for difficulty in difficulties:
            print(f"\n验证CTDE模式 - {difficulty}难度...")
            success = validator.run_validation(difficulty, True)

            if not success:
                all_success = False
                print(f"❌ {difficulty}难度CTDE验证失败")
            else:
                print(f"✅ {difficulty}难度CTDE验证通过")

            validator.cleanup()
            validator = MSFSEnvironmentValidator()

        if all_success:
            print("\n🎉 所有CTDE模式验证均通过！")
            return 0
        else:
            print("\n⚠️  部分CTDE模式验证失败")
            return 1

    except Exception as e:
        print(f"\n💥 CTDE模式验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def run_role_emergence_validation():
    """运行角色涌现专项验证"""
    print("🎭 开始MSFS环境角色涌现专项验证...")

    validator = MSFSEnvironmentValidator()

    try:
        # 测试角色涌现重点配置
        from Env.MSFS.config import MSFSPresetConfigs

        # 创建角色涌现重点配置的环境
        role_config = MSFSPresetConfigs.role_emergence_focus()

        # 设置环境
        from Env.MSFS.env_msfs_ctde import MSFSCTDEWrapper

        # 创建配置字典，避免参数冲突
        config_dict = role_config.to_dict()
        config_dict.pop('difficulty', None)  # 移除difficulty避免冲突

        validator.env = MSFSCTDEWrapper(difficulty="normal", global_state_type="concat", **config_dict)

        print(f"使用角色涌现配置:")
        print(f"  - 专门化奖励: {role_config.specialization_reward}")
        print(f"  - 完成奖励: {role_config.finishing_reward}")
        print(f"  - 角色切换惩罚: {role_config.role_switch_penalty}")
        print(f"  - 专门化阈值: {role_config.specialization_threshold}")

        success = validator.run_validation("normal", True)

        if success:
            print("\n🎉 角色涌现专项验证完成！")
            return 0
        else:
            print("\n⚠️  角色涌现专项验证发现问题。")
            return 1

    except Exception as e:
        print(f"\n💥 角色涌现专项验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def run_efficiency_validation():
    """运行效率重点专项验证"""
    print("⚡ 开始MSFS环境效率重点专项验证...")

    validator = MSFSEnvironmentValidator()

    try:
        # 测试效率重点配置
        from Env.MSFS.config import MSFSPresetConfigs

        # 创建效率重点配置的环境
        efficiency_config = MSFSPresetConfigs.efficiency_focus()

        # 设置环境
        from Env.MSFS.env_msfs_ctde import MSFSCTDEWrapper

        # 创建配置字典，避免参数冲突
        config_dict = efficiency_config.to_dict()
        config_dict.pop('difficulty', None)  # 移除difficulty避免冲突

        validator.env = MSFSCTDEWrapper(difficulty="hard", global_state_type="concat", **config_dict)

        print(f"使用效率重点配置:")
        print(f"  - 时间惩罚: {efficiency_config.step_penalty}")
        print(f"  - 空闲惩罚: {efficiency_config.idle_penalty}")
        print(f"  - 最大步数: {efficiency_config.max_steps}")
        print(f"  - 简单订单价值: {efficiency_config.simple_order_value}")
        print(f"  - 复杂订单价值: {efficiency_config.complex_order_value}")

        success = validator.run_validation("hard", True)

        if success:
            print("\n🎉 效率重点专项验证完成！")
            return 0
        else:
            print("\n⚠️  效率重点专项验证发现问题。")
            return 1

    except Exception as e:
        print(f"\n💥 效率重点专项验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def run_curriculum_validation():
    """运行课程学习验证"""
    print("📚 开始MSFS环境课程学习验证...")

    validator = MSFSEnvironmentValidator()
    total_success = True

    try:
        # 测试所有课程学习阶段
        curriculum_configs = [
            ("Stage1", MSFSPresetConfigs.curriculum_stage1),
            ("Stage2", MSFSPresetConfigs.curriculum_stage2),
            ("Stage3", MSFSPresetConfigs.curriculum_stage3),
        ]

        for stage_name, config_func in curriculum_configs:
            print(f"\n验证课程学习阶段: {stage_name}")

            config = config_func()
            from Env.MSFS.env_msfs_ctde import MSFSCTDEWrapper

            # 创建配置字典，避免参数冲突
            config_dict = config.to_dict()
            config_dict.pop('difficulty', None)  # 移除difficulty避免冲突

            validator.env = MSFSCTDEWrapper(difficulty=config.difficulty, global_state_type="concat", **config_dict)

            print(f"  - 课程阶段: {config.curriculum_stage}")
            print(f"  - 难度: {config.difficulty}")
            print(f"  - 智能体数: {config.num_agents}")

            success = validator.run_validation(config.difficulty, True)

            if not success:
                total_success = False
                print(f"❌ {stage_name} 验证失败")
            else:
                print(f"✅ {stage_name} 验证通过")

            validator.cleanup()
            validator = MSFSEnvironmentValidator()

        print(f"\n{'='*60}")
        if total_success:
            print("🎉 所有课程学习阶段验证均通过！")
            return 0
        else:
            print("⚠️  部分课程学习阶段验证失败")
            return 1

    except Exception as e:
        print(f"\n💥 课程学习验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def print_usage():
    """打印使用说明"""
    print("MSFS环境验证程序")
    print("=" * 40)
    print("使用方法:")
    print("  python run_msfs_validation.py              # 快速验证（推荐）")
    print("  python run_msfs_validation.py --comprehensive  # 全面验证")
    print("  python run_msfs_validation.py --basic-only     # 仅基础模式")
    print("  python run_msfs_validation.py --ctde-only      # 仅CTDE模式")
    print("  python run_msfs_validation.py --role-emergence # 角色涌现专项")
    print("  python run_msfs_validation.py --efficiency     # 效率重点专项")
    print("  python run_msfs_validation.py --curriculum     # 课程学习验证")
    print("  python run_msfs_validation.py --help           # 显示帮助")
    print()
    print("验证内容:")
    print("  - 观测空间（24维）验证")
    print("  - 动作空间（8维）验证")
    print("  - 奖励机制验证")
    print("  - CTDE全局状态（42维）验证（CTDE模式）")
    print("  - 工作站系统验证")
    print("  - 订单系统验证")
    print("  - 专门化机制验证")
    print()
    print("退出码:")
    print("  0 - 验证通过")
    print("  1 - 验证失败或出现错误")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--help", "-h"]:
            print_usage()
            return 0
        elif arg == "--comprehensive":
            return run_comprehensive_validation()
        elif arg == "--basic-only":
            return run_basic_validation()
        elif arg == "--ctde-only":
            return run_ctde_validation()
        elif arg == "--role-emergence":
            return run_role_emergence_validation()
        elif arg == "--efficiency":
            return run_efficiency_validation()
        elif arg == "--curriculum":
            return run_curriculum_validation()
        else:
            print(f"未知参数: {arg}")
            print_usage()
            return 1
    else:
        return run_quick_validation()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)