#!/usr/bin/env python3
"""
DEM环境验证程序运行脚本

使用方法:
    python run_dem_validation.py
    python run_dem_validation.py --comprehensive
    python run_dem_validation.py --basic-only

作者: Claude Code
日期: 2025-01-08
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verify_dem_environment import DEMEnvironmentValidator


def run_quick_validation():
    """运行快速验证"""
    print("🚀 开始DEM环境快速验证...")

    validator = DEMEnvironmentValidator()

    try:
        # 只验证normal难度的CTDE模式，这是最常用的配置
        success = validator.run_validation("normal", True)

        if success:
            print("\n🎉 验证完成！DEM环境实现与文档完全一致。")
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
    print("🔬 开始DEM环境全面验证...")

    test_configs = [
        ("normal", True, "CTDE模式"),
        ("normal", False, "基础模式"),
        ("easy", True, "简单难度CTDE"),
        ("hard", True, "困难难度CTDE"),
    ]

    validator = DEMEnvironmentValidator()
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
            validator = DEMEnvironmentValidator()

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
    print("🔧 开始DEM环境基础模式验证...")

    validator = DEMEnvironmentValidator()

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
    print("🧠 开始DEM环境CTDE模式验证...")

    validator = DEMEnvironmentValidator()

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
            validator = DEMEnvironmentValidator()

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


def print_usage():
    """打印使用说明"""
    print("DEM环境验证程序")
    print("=" * 40)
    print("使用方法:")
    print("  python run_dem_validation.py              # 快速验证（推荐）")
    print("  python run_dem_validation.py --comprehensive  # 全面验证")
    print("  python run_dem_validation.py --basic-only     # 仅基础模式")
    print("  python run_dem_validation.py --ctde-only      # 仅CTDE模式")
    print("  python run_dem_validation.py --help           # 显示帮助")
    print()
    print("验证内容:")
    print("  - 观测空间（52维）验证")
    print("  - 动作空间（10维）验证")
    print("  - 奖励机制验证")
    print("  - CTDE全局状态（41维）验证（CTDE模式）")
    print("  - 地形系统验证")
    print("  - VIP行为验证")
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
        else:
            print(f"未知参数: {arg}")
            print_usage()
            return 1
    else:
        return run_quick_validation()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)