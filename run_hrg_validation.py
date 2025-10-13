#!/usr/bin/env python3
"""
HRG环境验证程序运行脚本

使用方法:
    python run_hrg_validation.py

作者: Claude Code
日期: 2025-01-07
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verify_hrg_environment import HRGEnvironmentValidator


def run_quick_validation():
    """运行快速验证"""
    print("🚀 开始HRG环境快速验证...")

    validator = HRGEnvironmentValidator()

    try:
        # 只验证normal难度，这是最常用的配置
        success = validator.run_validation("normal")

        if success:
            print("\n🎉 验证完成！HRG环境实现与文档完全一致。")
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
    """运行全面验证（测试所有难度）"""
    print("🔬 开始HRG环境全面验证...")

    difficulties = ["easy", "normal", "hard"]
    validator = HRGEnvironmentValidator()

    total_success = True

    try:
        for difficulty in difficulties:
            print(f"\n{'='*60}")
            print(f"验证难度: {difficulty}")
            print('='*60)

            success = validator.run_validation(difficulty)

            if not success:
                total_success = False
                print(f"❌ {difficulty} 难度验证失败")
            else:
                print(f"✅ {difficulty} 难度验证通过")

            validator.cleanup()

            # 重新创建环境用于下一个难度测试
            validator = HRGEnvironmentValidator()

        print(f"\n{'='*60}")
        if total_success:
            print("🎉 所有难度验证均通过！")
            return 0
        else:
            print("⚠️  部分难度验证失败")
            return 1

    except Exception as e:
        print(f"\n💥 验证过程中出现异常: {e}")
        return 1
    finally:
        validator.cleanup()


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        return run_comprehensive_validation()
    else:
        return run_quick_validation()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)