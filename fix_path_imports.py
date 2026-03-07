#!/usr/bin/env python3
"""批量修复Path导入问题"""

import re
from pathlib import Path

files_to_fix = [
    'backtest_engine.py',
    'enhanced_data_layer.py',
    'macro_terminal_tushare.py',
    'run_backtest.py',
    'stock_universe.py'
]

for filename in files_to_fix:
    filepath = Path(filename)
    if not filepath.exists():
        print(f"❌ 文件不存在: {filename}")
        continue
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 检查是否已导入Path
    if 'from pathlib import Path' in content or 'import pathlib' in content:
        print(f"✅ {filename}: 已导入Path")
        continue
    
    # 修复: 在文件开头添加Path导入
    # 找到第一个import语句
    lines = content.split('\n')
    import_idx = None
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_idx = i
            break
    
    if import_idx is not None:
        # 在第一个import前插入Path导入
        lines.insert(import_idx, 'from pathlib import Path')
        new_content = '\n'.join(lines)
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print(f"✅ {filename}: 已修复")
    else:
        print(f"⚠️ {filename}: 未找到import语句")

print("\n修复完成!")
