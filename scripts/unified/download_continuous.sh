#!/bin/bash
# 持续下载脚本 - 直到所有股票下载完成

cd /root/.openclaw/workspace/myQuant/scripts/unified

echo "========================================"
echo "开始持续下载股票数据"
echo "开始时间: $(date)"
echo "========================================"

while true; do
    echo ""
    echo "[$(date)] 开始新批次下载..."
    
    python3 -c "
import sys
sys.path.insert(0, '/root/.openclaw/workspace/myQuant/scripts/unified')
from stock_database import StockDatabase
import time

db = StockDatabase()

# 获取当前统计
stats = db.get_statistics()
remaining = stats['total_stocks'] - stats['stocks_with_data']

if remaining <= 0:
    print('✅ 所有股票下载完成！')
    exit(0)

print(f'剩余 {remaining} 只股票需要下载')

# 下载一批
progress = db.batch_download(
    start_date='20200101',
    end_date='20250226',
    max_workers=5,
    batch_size=50
)

print(f'本批次: 成功 {progress.completed_stocks}, 失败 {len(progress.failed_stocks)}')

# 更新统计
stats = db.get_statistics()
pct = stats['stocks_with_data']/stats['total_stocks']*100
print(f'累计: {stats[\"stocks_with_data\"]} 只 ({pct:.1f}%), {stats[\"total_records\"]:,} 条记录')
"
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] 批次完成，暂停30秒..."
        sleep 30
    else
        echo "[$(date)] 出现错误，暂停60秒后重试..."
        sleep 60
    fi
done

echo ""
echo "========================================"
echo "下载完成！"
echo "结束时间: $(date)"
echo "========================================"
