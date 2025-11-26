# historical_backtest.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Any
from production_trading_system import ProductionTradingSystem

class HistoricalBacktest:
    """历史数据回测系统"""
    
    def __init__(self):
        self.trading_system = ProductionTradingSystem()
        self.backtest_results = {}
    
    def load_historical_data(self, symbol: str, start_date: str, end_date: str):
        """加载历史数据（模拟真实数据）"""
        print(f"📊 加载 {symbol} 历史数据: {start_date} 到 {end_date}")
        
        # 生成模拟的历史数据（实际项目中应该从API或数据库获取）
        return self._generate_historical_data(symbol, start_date, end_date)
    
    def _generate_historical_data(self, symbol: str, start_date: str, end_date: str):
        """生成模拟的历史数据（基于真实市场特征）"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        
        # 确保数据足够长（至少100天）
        if days < 100:
            days = 100
            print(f"⚠️ 数据周期过短，扩展到 {days} 天以确保策略计算")
        
        # 基础价格（基于不同币种的历史价格）
        base_prices = {
            'BTC-USDT': 45000,  # 2024年初大致价格
            'ETH-USDT': 2500,
            'ADA-USDT': 0.4
        }
        base_price = base_prices.get(symbol, 100)
        
        # 生成每日价格数据
        dates = [start_dt + timedelta(days=i) for i in range(days)]
        prices = [base_price]
        
        # 模拟2024年真实市场波动
        for i in range(1, len(dates)):
            # 2024年加密货币市场特征：总体上涨但波动较大
            if symbol == 'BTC-USDT':
                # BTC 2024年特征：从4.5万到6.7万左右
                trend = 0.001  # 轻微上涨趋势
                volatility = 0.03
            elif symbol == 'ETH-USDT':
                # ETH 2024年特征：从2.5k到3.8k左右
                trend = 0.0008
                volatility = 0.035
            else:
                trend = 0.0005
                volatility = 0.04
            
            # 生成价格变化
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.3))  # 防止价格归零
        
        # 创建DataFrame
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000000, 50000000) for _ in prices]
        })
        
        data.set_index('timestamp', inplace=True)
        print(f"✅ 生成 {len(data)} 天历史数据，价格范围: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        return data
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, initial_balance: float = 10000.0):
        """运行历史回测"""
        print(f"\n🎯 开始回测: {symbol} ({start_date} 到 {end_date})")
        
        # 加载历史数据
        historical_data = self.load_historical_data(symbol, start_date, end_date)
        
        # 确保数据足够长
        if len(historical_data) < 50:
            print(f"⚠️ 数据量不足，无法进行有效回测")
            return {
                'symbol': symbol,
                'initial_balance': initial_balance,
                'final_value': initial_balance,
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'trade_history': [],
                'portfolio_values': []
            }
        
        # 初始化策略（使用前半段数据）
        split_idx = len(historical_data) // 2
        training_data = historical_data.iloc[:split_idx]
        testing_data = historical_data.iloc[split_idx:]
        
        print(f"📊 数据分割: 训练集 {len(training_data)} 天, 测试集 {len(testing_data)} 天")
        
        # 使用训练数据优化策略
        self.trading_system.initialize_optimized_strategies(training_data)
        
        # 在测试数据上运行回测
        results = self._run_trading_simulation(testing_data, initial_balance, symbol)
        
        # 保存结果
        self.backtest_results[symbol] = results
        return results
    
    def _run_trading_simulation(self, data: pd.DataFrame, initial_balance: float, symbol: str):
        """运行交易模拟"""
        balance = initial_balance
        positions = {}
        trade_history = []
        portfolio_values = []
        
        # 按日期循环（模拟每日交易）
        for date, daily_data in data.groupby(data.index.date):
            # 获取当天的最后一条数据作为收盘价
            if len(daily_data) == 0:
                continue
                
            current_data = daily_data.iloc[-1:].copy()
            current_price = current_data['close'].iloc[0]
            
            # 获取交易决策 - 使用更多数据点
            lookback_data = data[data.index.date <= date].tail(50)  # 使用最近50个数据点
            if len(lookback_data) < 20:  # 确保数据足够
                continue
                
            decision = self.trading_system.process_market_data(lookback_data)
            
            # 执行交易逻辑
            if 'error' not in decision:
                action = decision['action']
                confidence = decision['confidence']
                
                # 降低置信度阈值以便测试
                if action == 'BUY' and confidence > 0.3 and balance > 100:
                    # 执行买入
                    position_value = min(balance * 0.1, 1000)  # 每次买入最多1000美元
                    quantity = position_value / current_price
                    
                    trade = {
                        'date': date,
                        'action': 'BUY',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': current_price,
                        'value': position_value
                    }
                    
                    if symbol in positions:
                        # 平均成本法
                        old_qty = positions[symbol]['quantity']
                        old_cost = old_qty * positions[symbol]['avg_price']
                        new_cost = old_cost + position_value
                        new_qty = old_qty + quantity
                        new_avg_price = new_cost / new_qty
                        
                        positions[symbol].update({
                            'quantity': new_qty,
                            'avg_price': new_avg_price
                        })
                    else:
                        positions[symbol] = {
                            'quantity': quantity,
                            'avg_price': current_price,
                            'entry_price': current_price
                        }
                    
                    balance -= position_value
                    trade_history.append(trade)
                    print(f"✅ {date}: 买入 {quantity:.6f} {symbol} @ ${current_price:,.2f}")
                    
                elif action == 'SELL' and confidence > 0.3 and symbol in positions:
                    # 执行卖出
                    position = positions[symbol]
                    quantity = position['quantity']
                    sell_value = quantity * current_price
                    cost_basis = quantity * position['avg_price']
                    profit_loss = sell_value - cost_basis
                    profit_loss_pct = (profit_loss / cost_basis) * 100
                    
                    trade = {
                        'date': date,
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': current_price,
                        'value': sell_value,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct
                    }
                    
                    balance += sell_value
                    del positions[symbol]
                    trade_history.append(trade)
                    
                    status = "盈利" if profit_loss > 0 else "亏损"
                    print(f"✅ {date}: 卖出 {quantity:.6f} {symbol} @ ${current_price:,.2f} | {status}: ${profit_loss:+.2f} ({profit_loss_pct:+.1f}%)")
            
            # 记录投资组合价值
            portfolio_value = balance
            for pos_symbol, position in positions.items():
                portfolio_value += position['quantity'] * current_price
            
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': balance,
                'positions': len(positions)
            })
        
        # 计算回测结果
        final_value = portfolio_values[-1]['value'] if portfolio_values else initial_balance
        total_return = (final_value - initial_balance) / initial_balance * 100
        
        # 计算胜率
        sell_trades = [t for t in trade_history if t['action'] == 'SELL']
        profitable_trades = len([t for t in sell_trades if t.get('profit_loss', 0) > 0])
        win_rate = (profitable_trades / len(sell_trades)) * 100 if sell_trades else 0
        
        return {
            'symbol': symbol,
            'initial_balance': initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trade_history),
            'win_rate': win_rate,
            'trade_history': trade_history[-10:],  # 最后10笔交易
            'portfolio_values': portfolio_values
        }
    
    def generate_backtest_report(self):
        """生成回测报告"""
        print(f"\n{'='*60}")
        print(f"📊 HISTORICAL BACKTEST REPORT")
        print(f"{'='*60}")
        
        for symbol, results in self.backtest_results.items():
            print(f"\n🎯 {symbol} 回测结果:")
            print(f"   初始资金: ${results['initial_balance']:,.2f}")
            print(f"   最终资金: ${results['final_value']:,.2f}")
            print(f"   总收益率: {results['total_return']:+.2f}%")
            print(f"   总交易次数: {results['total_trades']}")
            print(f"   胜率: {results['win_rate']:.1f}%")
            
            # 显示最近交易
            if results['trade_history']:
                print(f"\n   最近交易:")
                for trade in results['trade_history'][-3:]:
                    action = trade['action']
                    if action == 'SELL':
                        pnl = trade.get('profit_loss', 0)
                        pnl_pct = trade.get('profit_loss_pct', 0)
                        print(f"     {trade['date']}: {action} @ ${trade['price']:,.2f} | PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                    else:
                        print(f"     {trade['date']}: {action} @ ${trade['price']:,.2f}")

def run_comprehensive_backtest():
    """运行全面回测"""
    print("🚀 开始全面历史回测...")
    
    backtester = HistoricalBacktest()
    
    # 测试多个币种
    test_cases = [
        ('BTC-USDT', '2024-01-01', '2024-12-31'),
        ('ETH-USDT', '2024-01-01', '2024-12-31'),
        # ('ADA-USDT', '2024-01-01', '2024-12-31')  # 可以取消注释测试更多币种
    ]
    
    for symbol, start_date, end_date in test_cases:
        results = backtester.run_backtest(symbol, start_date, end_date, initial_balance=10000.0)
    
    # 生成综合报告
    backtester.generate_backtest_report()
    
    return backtester

if __name__ == "__main__":
    run_comprehensive_backtest()