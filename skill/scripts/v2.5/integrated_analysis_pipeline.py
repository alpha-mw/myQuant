"""
集成V2.5数据层的端到端分析流水线
将一手数据驱动的量化分析与V2.4的LLM增强分析框架深度融合
"""

import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5/data_service')
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.4')

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import os

from data_api import DataAPI


class IntegratedAnalysisPipeline:
    """集成分析流水线"""
    
    def __init__(self, tushare_token: Optional[str] = None):
        """
        初始化分析流水线
        
        Args:
            tushare_token: Tushare Pro的token
        """
        self.data_api = DataAPI(tushare_token=tushare_token)
        self.report_dir = '/home/ubuntu/quant_analysis_reports_v2.5'
        os.makedirs(self.report_dir, exist_ok=True)
    
    def analyze_stock(self, stock_code: str, stock_name: Optional[str] = None) -> Dict[str, Any]:
        """
        对单只股票进行完整的量化分析
        
        Args:
            stock_code: 股票代码（如 600519 或 600519.SH）
            stock_name: 股票名称（可选）
            
        Returns:
            Dict: 分析结果
        """
        # 标准化股票代码
        stock_code = self.data_api.normalize_stock_code(stock_code)
        
        print(f"\n{'='*60}")
        print(f"开始分析: {stock_name or stock_code}")
        print(f"{'='*60}\n")
        
        result = {
            'stock_code': stock_code,
            'stock_name': stock_name or stock_code,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 1. 获取价格数据
        print("1. 获取价格数据...")
        try:
            price_data = self.data_api.get_stock_price(stock_code, days=250, adjust='qfq')
            result['price_data'] = price_data
            result['price_data_available'] = True
            print(f"   ✅ 成功获取 {len(price_data)} 条价格数据")
            
            # 计算基本统计指标
            latest_price = price_data.iloc[0]['close']
            result['latest_price'] = latest_price
            result['price_change_30d'] = ((price_data.iloc[0]['close'] / price_data.iloc[min(29, len(price_data)-1)]['close']) - 1) * 100
            result['price_change_60d'] = ((price_data.iloc[0]['close'] / price_data.iloc[min(59, len(price_data)-1)]['close']) - 1) * 100
            result['price_change_120d'] = ((price_data.iloc[0]['close'] / price_data.iloc[min(119, len(price_data)-1)]['close']) - 1) * 100
            
            print(f"   最新价格: {latest_price:.2f}")
            print(f"   30日涨跌幅: {result['price_change_30d']:.2f}%")
            print(f"   60日涨跌幅: {result['price_change_60d']:.2f}%")
            print(f"   120日涨跌幅: {result['price_change_120d']:.2f}%")
            
        except Exception as e:
            print(f"   ❌ 价格数据获取失败: {e}")
            result['price_data_available'] = False
        
        # 2. 获取财务数据
        print("\n2. 获取财务数据...")
        try:
            financial_data = self.data_api.get_stock_financial(stock_code)
            result['financial_data'] = financial_data
            result['financial_data_available'] = not financial_data.empty
            
            if not financial_data.empty:
                print(f"   ✅ 成功获取财务数据")
                # 提取关键财务指标
                if 'roe' in financial_data.columns:
                    result['roe'] = financial_data.iloc[0]['roe']
                    print(f"   ROE: {result['roe']:.2f}%")
                if 'roa' in financial_data.columns:
                    result['roa'] = financial_data.iloc[0]['roa']
                    print(f"   ROA: {result['roa']:.2f}%")
                if 'debt_to_assets' in financial_data.columns:
                    result['debt_to_assets'] = financial_data.iloc[0]['debt_to_assets']
                    print(f"   资产负债率: {result['debt_to_assets']:.2f}%")
            else:
                print(f"   ⚠️  财务数据为空")
                
        except Exception as e:
            print(f"   ❌ 财务数据获取失败: {e}")
            result['financial_data_available'] = False
        
        # 3. 计算技术指标
        print("\n3. 计算技术指标...")
        if result.get('price_data_available'):
            try:
                price_df = result['price_data'].copy()
                price_df = price_df.sort_values('trade_date')  # 按时间正序排列
                
                # 计算MA
                price_df['ma5'] = price_df['close'].rolling(window=5).mean()
                price_df['ma20'] = price_df['close'].rolling(window=20).mean()
                price_df['ma60'] = price_df['close'].rolling(window=60).mean()
                
                # 计算RSI
                delta = price_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                price_df['rsi'] = 100 - (100 / (1 + rs))
                
                result['technical_indicators'] = price_df
                
                latest = price_df.iloc[-1]
                result['ma5'] = latest['ma5']
                result['ma20'] = latest['ma20']
                result['ma60'] = latest['ma60']
                result['rsi'] = latest['rsi']
                
                print(f"   ✅ 技术指标计算完成")
                print(f"   MA5: {result['ma5']:.2f}")
                print(f"   MA20: {result['ma20']:.2f}")
                print(f"   MA60: {result['ma60']:.2f}")
                print(f"   RSI: {result['rsi']:.2f}")
                
            except Exception as e:
                print(f"   ❌ 技术指标计算失败: {e}")
        
        # 4. 生成分析报告
        print("\n4. 生成分析报告...")
        try:
            report = self._generate_report(result)
            result['report'] = report
            
            # 保存报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"{self.report_dir}/{stock_code.replace('.', '_')}_{timestamp}_analysis.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            result['report_path'] = report_path
            print(f"   ✅ 报告已保存: {report_path}")
            
        except Exception as e:
            print(f"   ❌ 报告生成失败: {e}")
        
        return result
    
    def _generate_report(self, result: Dict[str, Any]) -> str:
        """生成分析报告"""
        
        report = f"""# {result['stock_name']} ({result['stock_code']}) 量化分析报告

**分析时间**: {result['analysis_time']}  
**数据来源**: Tushare Pro + AKShare（一手原始数据）

---

## 一、价格分析

"""
        
        if result.get('price_data_available'):
            report += f"""
**最新价格**: {result.get('latest_price', 'N/A'):.2f} 元

**价格走势**：
- 30日涨跌幅: {result.get('price_change_30d', 0):.2f}%
- 60日涨跌幅: {result.get('price_change_60d', 0):.2f}%
- 120日涨跌幅: {result.get('price_change_120d', 0):.2f}%

**技术指标**：
- MA5: {result.get('ma5', 'N/A'):.2f}
- MA20: {result.get('ma20', 'N/A'):.2f}
- MA60: {result.get('ma60', 'N/A'):.2f}
- RSI(14): {result.get('rsi', 'N/A'):.2f}

**技术面解读**：
"""
            # 简单的技术面分析
            latest_price = result.get('latest_price', 0)
            ma20 = result.get('ma20', 0)
            ma60 = result.get('ma60', 0)
            rsi = result.get('rsi', 50)
            
            if latest_price > ma20 > ma60:
                report += "- 价格位于MA20和MA60之上，短期和中期趋势向上\n"
            elif latest_price < ma20 < ma60:
                report += "- 价格位于MA20和MA60之下，短期和中期趋势向下\n"
            else:
                report += "- 价格处于均线纠缠状态，趋势不明朗\n"
            
            if rsi > 70:
                report += "- RSI超过70，可能处于超买状态\n"
            elif rsi < 30:
                report += "- RSI低于30，可能处于超卖状态\n"
            else:
                report += "- RSI处于正常区间\n"
        else:
            report += "\n⚠️ 价格数据不可用\n"
        
        report += "\n---\n\n## 二、基本面分析\n\n"
        
        if result.get('financial_data_available'):
            report += f"""
**财务指标**：
- ROE（净资产收益率）: {result.get('roe', 'N/A'):.2f}%
- ROA（总资产收益率）: {result.get('roa', 'N/A'):.2f}%
- 资产负债率: {result.get('debt_to_assets', 'N/A'):.2f}%

**基本面解读**：
"""
            roe = result.get('roe', 0)
            roa = result.get('roa', 0)
            debt_ratio = result.get('debt_to_assets', 0)
            
            if roe > 15:
                report += "- ROE超过15%，盈利能力较强\n"
            elif roe > 10:
                report += "- ROE在10-15%之间，盈利能力中等\n"
            else:
                report += "- ROE低于10%，盈利能力较弱\n"
            
            if debt_ratio < 40:
                report += "- 资产负债率低于40%，财务风险较低\n"
            elif debt_ratio < 60:
                report += "- 资产负债率在40-60%之间，财务风险适中\n"
            else:
                report += "- 资产负债率超过60%，财务风险较高\n"
        else:
            report += "\n⚠️ 财务数据不可用\n"
        
        report += "\n---\n\n## 三、综合评估\n\n"
        report += "**注意**：本报告基于一手原始数据进行量化分析，但仅供参考，不构成投资建议。投资决策需要结合更多因素，包括但不限于宏观经济环境、行业景气度、公司治理、市场情绪等。\n\n"
        report += "**数据质量说明**：\n"
        report += "- 价格数据：采用前复权（qfq）处理，已考虑分红送股对价格的影响\n"
        report += "- 财务数据：来自上市公司定期报告，具有滞后性\n"
        report += "- 数据来源：Tushare Pro（主）+ AKShare（辅）\n"
        
        return report
    
    def analyze_market(self) -> Dict[str, Any]:
        """分析整体市场环境（宏观数据）"""
        
        print(f"\n{'='*60}")
        print("市场宏观环境分析")
        print(f"{'='*60}\n")
        
        result = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 获取GDP数据
        print("1. 获取GDP数据...")
        try:
            gdp_data = self.data_api.get_gdp()
            result['gdp_data'] = gdp_data
            print(f"   ✅ 成功获取 {len(gdp_data)} 条GDP数据")
            print(gdp_data.tail())
        except Exception as e:
            print(f"   ❌ GDP数据获取失败: {e}")
        
        # 获取CPI数据
        print("\n2. 获取CPI数据...")
        try:
            cpi_data = self.data_api.get_cpi()
            result['cpi_data'] = cpi_data
            print(f"   ✅ 成功获取 {len(cpi_data)} 条CPI数据")
            print(cpi_data.tail())
        except Exception as e:
            print(f"   ❌ CPI数据获取失败: {e}")
        
        return result


if __name__ == '__main__':
    # 测试代码
    token = os.getenv('TUSHARE_TOKEN')
    pipeline = IntegratedAnalysisPipeline(tushare_token=token)
    
    # 测试单只股票分析
    print("\n" + "="*60)
    print("测试：单只股票完整分析")
    print("="*60)
    
    result = pipeline.analyze_stock('600519', '贵州茅台')
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    print(f"\n报告路径: {result.get('report_path')}")
    
    # 打印数据API统计
    print("\n数据API统计:")
    stats = pipeline.data_api.get_stats()
    print(f"Tushare调用次数: {stats['tushare_calls']}")
    print(f"AKShare调用次数: {stats['akshare_calls']}")
    print(f"缓存命中次数: {stats['cache_hits']}")
