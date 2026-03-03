#!/usr/bin/env python3
"""
Quant-Investor V7.0 系统状态评估脚本
评估当前系统完成度和质量
"""

import os
import sys
from pathlib import Path
import subprocess

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    full_path = PROJECT_ROOT / filepath
    exists = full_path.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {description}: {filepath}")
    return exists

def check_python_syntax(filepath):
    """检查Python语法"""
    full_path = PROJECT_ROOT / filepath
    if not full_path.exists():
        return False
    
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(full_path)],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False

def run_tests():
    """运行测试套件"""
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/unit/", "-v", "--tb=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout
        if "passed" in output:
            # 提取通过数量
            import re
            match = re.search(r'(\d+) passed', output)
            if match:
                return int(match.group(1))
        return 0
    except:
        return 0

def main():
    print("=" * 80)
    print("Quant-Investor V7.0 系统状态评估")
    print("=" * 80)
    
    score = 0
    max_score = 100
    
    # P0 检查
    print("\n📦 P0 - 工程化基础")
    print("-" * 40)
    
    p0_items = [
        ("requirements.txt", "依赖管理"),
        ("pyproject.toml", "项目配置"),
        ("tests/unit/test_data_layer.py", "数据层测试"),
        ("tests/unit/test_factor_layer.py", "因子层测试"),
        ("tests/unit/test_risk_management.py", "风控测试"),
        ("scripts/unified/logging_config.py", "日志配置"),
        ("scripts/unified/backtest_engine.py", "回测引擎"),
    ]
    
    p0_score = 0
    for filepath, desc in p0_items:
        if check_file_exists(filepath, desc):
            p0_score += 5
    
    print(f"\n  P0 得分: {p0_score}/35")
    score += p0_score
    
    # P1 检查
    print("\n🔧 P1 - 短期目标")
    print("-" * 40)
    
    p1_items = [
        ("scripts/unified/di_container.py", "依赖注入"),
        ("scripts/unified/cache_manager.py", "缓存管理"),
        ("scripts/unified/hyperparameter_tuner.py", "超参数调优"),
        ("scripts/unified/shap_explainer.py", "模型解释"),
    ]
    
    p1_score = 0
    for filepath, desc in p1_items:
        if check_file_exists(filepath, desc):
            p1_score += 5
    
    print(f"\n  P1 得分: {p1_score}/20")
    score += p1_score
    
    # 测试覆盖率
    print("\n🧪 测试状态")
    print("-" * 40)
    
    test_count = run_tests()
    if test_count > 0:
        print(f"  ✅ 单元测试: {test_count} 个通过")
        score += min(test_count * 2, 20)
    else:
        print(f"  ❌ 单元测试: 未运行或失败")
    
    # 代码质量
    print("\n📊 代码质量")
    print("-" * 40)
    
    key_files = [
        "scripts/unified/backtest_engine.py",
        "scripts/unified/v72_full_market_analysis.py",
    ]
    
    quality_score = 0
    for filepath in key_files:
        if check_python_syntax(filepath):
            quality_score += 5
    
    print(f"\n  代码质量得分: {quality_score}/10")
    score += quality_score
    
    # 文档完整性
    print("\n📚 文档状态")
    print("-" * 40)
    
    doc_files = ["README.md", "AGENTS.md", "MEMORY.md"]
    doc_score = 0
    for doc in doc_files:
        if (PROJECT_ROOT / doc).exists():
            doc_score += 5
    
    print(f"  文档得分: {doc_score}/15")
    score += doc_score
    
    # 总分
    print("\n" + "=" * 80)
    print(f"📈 综合评分: {score}/{max_score}")
    print("=" * 80)
    
    # 评级
    if score >= 85:
        grade = "A (优秀)"
    elif score >= 70:
        grade = "B (良好)"
    elif score >= 60:
        grade = "C (及格)"
    else:
        grade = "D (需改进)"
    
    print(f"\n评级: {grade}")
    
    # 建议
    print("\n💡 改进建议:")
    if p0_score < 35:
        print("  - 优先完成P0工程化基础建设")
    if test_count < 20:
        print("  - 增加单元测试覆盖率")
    if p1_score < 10:
        print("  - 开始P1架构优化")
    
    return score

if __name__ == "__main__":
    main()
