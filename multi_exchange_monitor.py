# multi_exchange_monitor.py
import asyncio
import pandas as pd
import numpy as np
import ccxt
from typing import Dict, List, Any
import logging

class SniperExchangeMonitor:
    """狙击手多交易所监控器 - 修复版"""
    
    def __init__(self):
        # 只使用币安
        self.exchanges = {
            'binance': ccxt.binance({
                'enableRateLimit': True,
            })
            
        }
        
        # 监控的币种 - 使用常见交易对
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 'SOL/USDT']
        self.volume_threshold = 2.5
        self.price_threshold = 0.02
        
        self.logger = logging.getLogger('ExchangeMonitor')
        self.logger.info("✅ 多交易所监控器初始化完成")
        
    async def monitor_volume_spike(self, symbol: str) -> List[Dict[str, Any]]:
        """监控成交量异常"""
        alerts = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                self.logger.debug(f"监控 {exchange_name} {symbol}...")
                
                # 获取K线数据
                ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=100)
                if len(ohlcv) < 20:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # 计算成交量异常
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].tail(20).mean()
                
                if avg_volume == 0:
                    continue
                    
                volume_ratio = current_volume / avg_volume
                
                # 计算价格突破
                current_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2]
                price_change = (current_close - prev_close) / prev_close
                
                if volume_ratio > self.volume_threshold and abs(price_change) > self.price_threshold:
                    alert = {
                        'exchange': exchange_name,
                        'symbol': symbol,
                        'volume_ratio': volume_ratio,
                        'price_change': price_change,
                        'current_price': current_close,
                        'timestamp': pd.Timestamp.now(),
                        'data': df
                    }
                    alerts.append(alert)
                    
                    self.logger.info(f"🎯 异常波动警报: {exchange_name} {symbol} "
                                  f"量比: {volume_ratio:.1f}x "
                                  f"涨幅: {price_change:.2%}")
                    
            except ccxt.BaseError as e:
                self.logger.debug(f"交易所 {exchange_name} {symbol} 暂时不可用: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"交易所 {exchange_name} {symbol} 监控错误: {e}")
                continue
        
        return alerts
    
    def get_market_data(self, symbol: str, exchange_name: str = 'binance', timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        """获取市场数据"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                self.logger.error(f"交易所 {exchange_name} 未找到")
                return pd.DataFrame()
                
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            self.logger.debug(f"获取 {exchange_name} {symbol} 数据成功: {len(df)} 条")
            return df
            
        except ccxt.BaseError as e:
            self.logger.error(f"获取 {exchange_name} {symbol} 数据失败: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取市场数据异常: {e}")
            return pd.DataFrame()

# 测试函数
async def test_exchange_monitor():
    """测试交易所监控器"""
    print("🧪 测试多交易所监控器...")
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = SniperExchangeMonitor()
    
    # 测试获取数据
    print("📊 测试数据获取...")
    test_symbols = ['BTC/USDT', 'ETH/USDT']
    test_exchanges = ['binance', 'okx']  # 使用更稳定的交易所
    
    for symbol in test_symbols:
        for exchange_name in test_exchanges:
            print(f"  获取 {exchange_name} {symbol}...")
            try:
                df = monitor.get_market_data(symbol, exchange_name, '5m', 10)
                if not df.empty:
                    print(f"  ✅ 成功获取 {len(df)} 条数据")
                    print(f"     最新价格: {df['close'].iloc[-1]:.2f}")
                else:
                    print(f"  ❌ 获取数据失败")
            except Exception as e:
                print(f"  ⚠️  {exchange_name} 错误: {e}")
    
    return monitor

if __name__ == "__main__":
    asyncio.run(test_exchange_monitor())