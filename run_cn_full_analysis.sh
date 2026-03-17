#!/bin/bash
# A股全市场批量分析 - 后台运行脚本

cd ${MYQUANT_REPO_ROOT}
source venv/bin/activate

export DEEPSEEK_API_KEY="sk-70e62000c8454fd18ab52a12fe468115"
export DASHSCOPE_API_KEY="sk-666ecf2b9c004dc2b192833520bc21a9"

LOG_FILE="cn_full_analysis_$(date +%Y%m%d_%H%M%S).log"
RESULTS_DIR="results/cn_analysis_full"

mkdir -p "$RESULTS_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "A股全市场批量分析启动: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 统计股票数量
echo "📊 待分析股票统计:" | tee -a "$LOG_FILE"
for dir in hs300 zz500 zz1000; do
  count=$(ls data/cn_market_full/$dir/*.csv 2>/dev/null | wc -l)
  echo "  $dir: $count 只" | tee -a "$LOG_FILE"
done
total=$(ls data/cn_market_full/*/*.csv 2>/dev/null | wc -l)
echo "  总计: $total 只" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[$(date)] 开始全市场五路并行分析并生成组合级交易计划..." | tee -a "$LOG_FILE"
python scripts/unified/cn_full_market_batch_analysis.py \
  --category all \
  --batch-size 30 \
  --capital 1000000 \
  --top-k 12 >> "$LOG_FILE" 2>&1
echo "[$(date)] 全市场分析与交易计划生成完成" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[$(date)] A股全市场分析全部完成!" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "结果目录: $RESULTS_DIR" | tee -a "$LOG_FILE"
