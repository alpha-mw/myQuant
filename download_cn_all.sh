#!/bin/bash
# A股全市场数据下载脚本 (后台运行)

cd /Users/maxwell/mySpace/myQuant
source venv/bin/activate

echo "========================================" >> download_cn.log
echo "开始下载A股全市场数据: $(date)" >> download_cn.log
echo "========================================" >> download_cn.log

# 下载中证500
echo "[$(date)] 下载中证500..." >> download_cn.log
python scripts/unified/download_full_cn_market.py --years 3 --category zz500 >> download_cn.log 2>&1

# 下载中证1000
echo "[$(date)] 下载中证1000..." >> download_cn.log
python scripts/unified/download_full_cn_market.py --years 3 --category zz1000 >> download_cn.log 2>&1

echo "[$(date)] 全部下载完成!" >> download_cn.log

# 生成报告
echo "" >> download_cn.log
echo "=== 下载统计 ===" >> download_cn.log
for dir in hs300 zz500 zz1000; do
  count=$(ls data/cn_market_full/$dir/*.csv 2>/dev/null | wc -l)
  echo "$dir: $count 只股票" >> download_cn.log
done
echo "总计: $(ls data/cn_market_full/*/*.csv 2>/dev/null | wc -l) 只股票" >> download_cn.log
echo "数据大小: $(du -sh data/cn_market_full/ 2>/dev/null | cut -f1)" >> download_cn.log
