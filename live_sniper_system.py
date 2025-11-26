# live_sniper_system.py
import asyncio
import ccxt
import pandas as pd
import logging
from typing import Dict, List, Any
from datetime import datetime
import os

class LiveSniperSystem:
    """实盘刺客交易系统"""
    
    def __init__(self, api_key: str = None, secret: str = None, testnet: bool = True):
        self.setup_logging()
        
        # 初始化交易所连接
        self.exchange = self._init_exchange(api_key, secret, testnet)
        
        # 导入策略组件
        from enhanced_sniper_detector import EnhancedSniperDetector
        from advanced_position_manager import AdvancedPositionManager
        
        self.signal_detector = EnhancedSniperDetector()
        self.position_manager = AdvancedPositionManager()
        
        # 交易参数
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        self.min_volume = 1000000  # 最小成交量
        self.max_positions = 3     # 最大同时持仓数
        
        self.active_positions = {}
        self.trade_history = []
        
        self.logger.info("🚀 实盘刺客系统初始化完成")
    
    def _init_exchange(self, api_key: str, secret: str, testnet: bool) -> ccxt.Exchange:
        """初始化交易所连接"""
        exchange = ccxt.binance({
            'apiKey': api_key or os.getenv('BINANCE_API_KEY'),
            'secret': secret or os.getenv('BINANCE_SECRET'),
            'sandbox': testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 使用合约交易
            }
        })
        
        if testnet:
            exchange.set_sandbox_mode(True)
            self.logger.info("🔧 使用币安测试网")
        else:
            self.logger.info("💰 使用币安实盘")
            
        return exchange
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_trading.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LiveSniper')
    
    async def fetch_market_data(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        """获取市场数据"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"获取 {symbol} 数据失败: {e}")
            return pd.DataFrame()
    
    async def monitor_markets(self):
        """监控市场"""
        self.logger.info("🔍 开始市场监控...")
        
        while True:
            try:
                for symbol in self.symbols:
                    # 跳过已有仓位的币种
                    if symbol in self.active_positions:
                        continue
                    
                    # 获取市场数据
                    df = await self.fetch_market_data(symbol)
                    if df.empty or len(df) < 50:
                        continue
                    
                    # 检查异常波动
                    alert = await self._check_volume_spike(symbol, df)
                    if alert:
                        # 信号确认
                        signal = self.signal_detector.confirm_sniper_signal(alert, df)
                        
                        if signal['confirmed']:
                            await self._execute_trade(signal, df)
                
                # 检查现有仓位的止损止盈
                await self._check_positions()
                
                # 等待下一轮监控
                await asyncio.sleep(30)  # 30秒间隔
                
            except Exception as e:
                self.logger.error(f"市场监控错误: {e}")
                await asyncio.sleep(60)
    
    async def _check_volume_spike(self, symbol: str, df: pd.DataFrame) -> Dict:
        """检查成交量异常"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        if avg_volume == 0:
            return None
            
        volume_ratio = current_volume / avg_volume
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        
        if volume_ratio > 3.0 and abs(price_change) > 0.025:
            return {
                'symbol': symbol,
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'current_price': df['close'].iloc[-1],
                'timestamp': datetime.now()
            }
        return None
    
    async def _execute_trade(self, signal: Dict, df: pd.DataFrame):
        """执行交易"""
        try:
            symbol = signal['symbol']
            
            # 计算市场条件
            market_conditions = {
                'volatility': np.std(np.diff(df['close'].tail(20)) / df['close'].tail(19))
            }
            
            # 计算仓位
            position_info = self.position_manager.calculate_dynamic_position(signal, market_conditions)
            
            # 检查资金和仓位限制
            if not self._can_open_position(symbol, position_info):
                return
            
            # 设置杠杆
            await self._set_leverage(symbol, position_info['leverage'])
            
            # 执行订单
            if signal['direction'] == 'LONG':
                order = await self.exchange.create_market_buy_order(symbol, position_info['quantity'])
            else:
                order = await self.exchange.create_market_sell_order(symbol, position_info['quantity'])
            
            # 记录交易
            trade_record = {
                'symbol': symbol,
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'quantity': position_info['quantity'],
                'leverage': position_info['leverage'],
                'position_size': position_info['position_size'],
                'stop_loss': position_info['stop_loss'],
                'take_profit': position_info['take_profit'],
                'order_id': order['id'],
                'timestamp': datetime.now(),
                'confidence': signal['confidence']
            }
            
            self.active_positions[symbol] = trade_record
            self.trade_history.append({**trade_record, 'action': 'OPEN'})
            
            self.logger.info(f"🎯 实盘开仓: {signal['direction']} {symbol} "
                          f"价格: {signal['entry_price']:.2f} "
                          f"仓位: ${position_info['position_size']:.0f} "
                          f"杠杆: {position_info['leverage']}x")
            
            # 设置止损止盈订单
            await self._place_stop_orders(symbol, signal['direction'], position_info)
            
        except Exception as e:
            self.logger.error(f"交易执行失败: {e}")
    
    async def _can_open_position(self, symbol: str, position_info: Dict) -> bool:
        """检查是否可以开仓"""
        # 检查是否已有该币种仓位
        if symbol in self.active_positions:
            return False
        
        # 检查最大仓位限制
        if len(self.active_positions) >= self.max_positions:
            return False
        
        # 检查资金是否足够
        balance = await self.exchange.fetch_balance()
        free_usdt = balance['USDT']['free']
        
        return free_usdt >= position_info['position_size'] * 1.1  # 留10%缓冲
    
    async def _set_leverage(self, symbol: str, leverage: int):
        """设置杠杆"""
        try:
            await self.exchange.set_leverage(leverage, symbol)
        except Exception as e:
            self.logger.warning(f"设置杠杆失败: {e}")
    
    async def _place_stop_orders(self, symbol: str, direction: str, position_info: Dict):
        """设置止损止盈订单"""
        try:
            # 这里需要根据交易所API设置止损止盈
            # 币安的具体实现会根据API版本有所不同
            pass
        except Exception as e:
            self.logger.error(f"设置止损止盈失败: {e}")
    
    async def _check_positions(self):
        """检查仓位状态"""
        try:
            positions = await self.exchange.fetch_positions()
            
            for pos in positions:
                symbol = pos['symbol']
                if symbol in self.active_positions and pos['contracts'] == 0:
                    # 仓位已平仓
                    await self._record_position_close(symbol, pos)
                    
        except Exception as e:
            self.logger.error(f"检查仓位失败: {e}")
    
    async def _record_position_close(self, symbol: str, position_data: Dict):
        """记录仓位平仓"""
        if symbol in self.active_positions:
            trade = self.active_positions[symbol]
            pnl = position_data.get('unrealizedPnl', 0)
            
            self.trade_history.append({
                'action': 'CLOSE',
                'symbol': symbol,
                'exit_time': datetime.now(),
                'exit_price': position_data.get('markPrice', 0),
                'pnl': pnl,
                'reason': 'MANUAL'  # 或其他平仓原因
            })
            
            del self.active_positions[symbol]
            
            status = "盈利" if pnl > 0 else "亏损"
            self.logger.info(f"💸 实盘平仓: {symbol} | {status}: ${pnl:+.2f}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history if t['action'] == 'CLOSE')
        
        return {
            'active_positions': len(self.active_positions),
            'total_trades': len([t for t in self.trade_history if t['action'] == 'CLOSE']),
            'total_pnl': total_pnl,
            'current_time': datetime.now().isoformat()
        }

# 启动函数
async def start_live_trading(api_key: str = None, secret: str = None, testnet: bool = True):
    """启动实盘交易"""
    print("🚀 启动刺客实盘交易系统...")
    print("⚠️  请确保已设置币安API密钥")
    
    sniper = LiveSniperSystem(api_key, secret, testnet)
    
    try:
        # 测试连接
        balance = await sniper.exchange.fetch_balance()
        print(f"✅ 连接成功! 余额: {balance['USDT']['free']:.2f} USDT")
        
        # 开始监控
        await sniper.monitor_markets()
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    # 从环境变量获取API密钥或直接传入
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET')
    
    asyncio.run(start_live_trading(api_key, secret, testnet=True))