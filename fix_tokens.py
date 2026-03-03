#!/usr/bin/env python3
"""
批量修复硬编码token
"""

import re
from pathlib import Path

PROJECT_ROOT = Path("/root/.openclaw/workspace/myQuant")

# 需要修复的文件列表
files_to_fix = [
    "scripts/unified/enhanced_data_layer.py",
    "scripts/unified/quant_investor_v71.py",
    "scripts/unified/quant_investor_v7.py",
    "scripts/unified/stock_universe.py",
]

# 修复模式
old_pattern = r"TUSHARE_TOKEN\s*=\s*os\.environ\.get\('TUSHARE_TOKEN',\s*'[^']+'\)"
new_code = "TUSHARE_TOKEN = config.TUSHARE_TOKEN"

old_pattern2 = r'TUSHARE_TOKEN\s*=\s*"[^"]+"'
new_code2 = "TUSHARE_TOKEN = config.TUSHARE_TOKEN"

for filepath in files_to_fix:
    full_path = PROJECT_ROOT / filepath
    if not full_path.exists():
        print(f"❌ 文件不存在: {filepath}")
        continue
    
    try:
        with open(full_path, 'r') as f:
            content = f.read()
        
        # 检查是否已经修复
        if 'from config import config' in content:
            print(f"✅ 已修复: {filepath}")
            continue
        
        # 添加config导入
        if 'import sys' in content and 'sys.path.insert' in content:
            # 在sys.path.insert后添加
            content = content.replace(
                'sys.path.insert(0,',
                'sys.path.insert(0, str(Path(__file__).parent))\nfrom config import config\nsys.path.insert(0,'
            )
        else:
            # 在import pandas后添加
            content = content.replace(
                'import pandas as pd',
                'from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).parent))\nfrom config import config\n\nimport pandas as pd'
            )
        
        # 替换token定义
        content = re.sub(old_pattern, new_code, content)
        content = re.sub(old_pattern2, new_code2, content)
        
        with open(full_path, 'w') as f:
            f.write(content)
        
        print(f"✅ 已修复: {filepath}")
        
    except Exception as e:
        print(f"❌ 修复失败 {filepath}: {e}")

print("\n修复完成！请检查文件是否正确修改。")
