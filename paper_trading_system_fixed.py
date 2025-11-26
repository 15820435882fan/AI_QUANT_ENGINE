# paper_trading_system_fixed.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from production_trading_system import ProductionTradingSystem
from real_market_data import RealMarketData

class PaperTradingSystemFixed:
    """修复版实盘模拟交易系统"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.trading_system = ProductionTradingSystem()
        self.market_data = RealMarketData()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.portfolio_value = []
        self.setup_trading()
    
    def setup_trading(self):
        """设置交易参数"""
        self.symbol = 'BTC-USDT'
        self.position_size = 0.1  # 10%仓位
        self.stop_loss = 0.05     # 5%止损
        self.take_profit = 0.10   # 10%止盈
        
    def initialize_strategies(self, historical_days: int = 100):
        """初始化策略（关键修复）"""
        print("🔧 初始化交易策略...")
        
        # 获取足够的历史数据用于策略优化
        historical_data = self.market_data.get_binance_data(
            self.symbol, 
            limit=historical_days * 20  # 假设每天20个5分钟K线
        )
        
        # 初始化优化策略
        self.trading_system.initialize_optimized_strategies(historical_data)
        
        print(f"✅ 策略初始化完成，使用 {len(historical_data)} 条历史数据")
    
    def run_paper_trading(self, days: int = 30):
        """运行模拟交易"""
        print(f"📈 开始 {days} 天模拟交易...")
        print(f"初始资金: ${self.initial_balance:,.2f}")
        
        # 首先初始化策略
        self.initialize_strategies(historical_days=30)
        
        # 模拟多天交易
        for day in range(1, days + 1):
            print(f"\n--- 第 {day} 天 ---")
            
            try:
                # 获取当日数据（更多数据点）
                daily_data = self.market_data.get_binance_data(self.symbol, limit=96)  # 24小时 * 4个15分钟K线
                
                # 处理交易决策
                self.process_daily_trading(daily_data, day)
                
                # 记录投资组合价值
                self.record_portfolio_value(day)
                
                # 显示当日总结
                self.daily_summary(day)
                
            except Exception as e:
                print(f"❌ 第 {day} 天交易出错: {e}")
                continue
        
        # 生成最终报告
        self.generate_final_report()
    
    def process_daily_trading(self, data: pd.DataFrame, day: int):
        """处理每日交易"""
        # 获取交易决策
        decision = self.trading_system.process_market_data(data)
        
        if 'error' in decision:
            print(f"⚠️ 决策错误: {decision['error']}")
            return
        
        action = decision['action']
        confidence = decision['confidence']
        
        print(f"🎯 交易信号: {action} (置信度: {confidence:.2f})")
        
        # 执行交易逻辑（降低置信度阈值以便测试）
        if action == 'BUY' and confidence > 0.3:
            self.execute_buy(data, decision)
        elif action == 'SELL' and confidence > 0.3:
            self.execute_sell(data, decision)
        else:
            print(f"📊 信号强度不足，保持观望 (需要 > 0.3)")
    
    def execute_buy(self, data: pd.DataFrame, decision: dict):
        """执行买入操作"""
        current_price = data['close'].iloc[-1]
        position_value = self.balance * self.position_size
        
        if position_value < 10:  # 最小交易金额
            print("💰 资金不足，无法买入")
            return
        
        # 计算买入数量
        quantity = position_value / current_price
        
        # 记录交易
        trade = {
            'timestamp': datetime.now(),
            'action': 'BUY',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': current_price,
            'value': position_value,
            'confidence': decision['confidence']
        }
        
        # 更新仓位和资金
        if self.symbol in self.positions:
            # 平均成本法
            old_quantity = self.positions[self.symbol]['quantity']
            old_value = old_quantity * self.positions[self.symbol]['avg_price']
            new_value = old_value + position_value
            new_quantity = old_quantity + quantity
            new_avg_price = new_value / new_quantity
            
            self.positions[self.symbol].update({
                'quantity': new_quantity,
                'avg_price': new_avg_price
            })
        else:
            self.positions[self.symbol] = {
                'quantity': quantity,
                'avg_price': current_price,
                'entry_price': current_price
            }
        
        self.balance -= position_value
        self.trade_history.append(trade)
        
        print(f"✅ 买入 {quantity:.6f} {self.symbol} @ ${current_price:,.2f}")
        print(f"💰 花费: ${position_value:,.2f}, 剩余资金: ${self.balance:,.2f}")
    
    def execute_sell(self, data: pd.DataFrame, decision: dict):
        """执行卖出操作"""
        if self.symbol not in self.positions:
            print("📭 无持仓可卖出")
            return
        
        current_price = data['close'].iloc[-1]
        position = self.positions[self.symbol]
        quantity = position['quantity']
        
        # 计算卖出价值
        sell_value = quantity * current_price
        cost_basis = quantity * position['avg_price']
        profit_loss = sell_value - cost_basis
        profit_loss_pct = (profit_loss / cost_basis) * 100
        
        # 记录交易
        trade = {
            'timestamp': datetime.now(),
            'action': 'SELL',
            'symbol': self.symbol,
            'quantity': quantity,
            'price': current_price,
            'value': sell_value,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'confidence': decision['confidence']
        }
        
        # 更新资金和清空仓位
        self.balance += sell_value
        del self.positions[self.symbol]
        self.trade_history.append(trade)
        
        status = "盈利" if profit_loss > 0 else "亏损"
        print(f"✅ 卖出 {quantity:.6f} {self.symbol} @ ${current_price:,.2f}")
        print(f"💰 {status}: ${profit_loss:+.2f} ({profit_loss_pct:+.2f}%)")
        print(f"💰 当前资金: ${self.balance:,.2f}")
    
    def record_portfolio_value(self, day: int):
        """记录投资组合价值"""
        total_value = self.balance
        
        # 计算持仓价值
        for symbol, position in self.positions.items():
            # 使用最近价格估算持仓价值
            recent_data = self.market_data.get_binance_data(symbol, limit=1)
            current_price = recent_data['close'].iloc[-1]
            position_value = position['quantity'] * current_price
            total_value += position_value
        
        self.portfolio_value.append({
            'day': day,
            'total_value': total_value,
            'cash': self.balance,
            'positions_value': total_value - self.balance
        })
    
    def daily_summary(self, day: int):
        """每日总结"""
        if self.portfolio_value:
            current_value = self.portfolio_value[-1]['total_value']
            total_return = ((current_value - self.initial_balance) / self.initial_balance) * 100
            
            print(f"📊 第 {day} 天总结:")
            print(f"  投资组合价值: ${current_value:,.2f}")
            print(f"  总收益: {total_return:+.2f}%")
            print(f"  持仓数量: {len(self.positions)}")
            print(f"  交易次数: {len(self.trade_history)}")
    
    def generate_final_report(self):
        """生成最终报告"""
        if not self.portfolio_value:
            return
        
        final_value = self.portfolio_value[-1]['total_value']
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        total_trades = len(self.trade_history)
        
        # 计算胜率
        profitable_trades = len([t for t in self.trade_history 
                               if t.get('profit_loss', 0) > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"🎉 模拟交易最终报告")
        print(f"{'='*50}")
        print(f"📈 初始资金: ${self.initial_balance:,.2f}")
        print(f"💰 最终资金: ${final_value:,.2f}")
        print(f"📊 总收益率: {total_return:+.2f}%")
        print(f"🔄 总交易次数: {total_trades}")
        print(f"🎯 交易胜率: {win_rate:.1f}%")
        print(f"📦 最终持仓: {len(self.positions)} 个")
        
        # 显示交易历史
        if self.trade_history:
            print(f"\n📋 最近交易:")
            for i, trade in enumerate(self.trade_history[-5:], 1):  # 显示最后5笔交易
                action = trade['action']
                symbol = trade['symbol']
                price = trade['price']
                if action == 'SELL':
                    pnl = trade.get('profit_loss', 0)
                    pnl_pct = trade.get('profit_loss_pct', 0)
                    print(f"  {i}. {action} {symbol} @ ${price:,.2f} | PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                else:
                    print(f"  {i}. {action} {symbol} @ ${price:,.2f}")

def test_paper_trading_fixed():
    """测试修复版模拟交易"""
    print("🧪 测试修复版模拟交易系统...")
    
    # 运行14天模拟交易（更长时间）
    paper_trader = PaperTradingSystemFixed(initial_balance=5000.0)
    paper_trader.run_paper_trading(days=14)
    
    return paper_trader

if __name__ == "__main__":
    test_paper_trading_fixed()