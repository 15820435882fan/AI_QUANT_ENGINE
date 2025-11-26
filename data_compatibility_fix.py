# data_compatibility_fix.py
#!/usr/bin/env python3

class DataCompatibility:
    """数据兼容性修复工具"""
    
    @staticmethod
    def fix_market_data(market_data):
        """修复市场数据对象"""
        if not hasattr(market_data, 'data_type'):
            market_data.data_type = 'ohlc'
        
        # 确保有必要的属性
        if not hasattr(market_data, 'high'):
            if hasattr(market_data, 'data') and len(market_data.data) > 2:
                market_data.high = market_data.data[2]
            else:
                market_data.high = getattr(market_data, 'close', 0) * 1.005
        
        if not hasattr(market_data, 'low'):
            if hasattr(market_data, 'data') and len(market_data.data) > 3:
                market_data.low = market_data.data[3]
            else:
                market_data.low = getattr(market_data, 'close', 0) * 0.995
        
        if not hasattr(market_data, 'open'):
            if hasattr(market_data, 'data') and len(market_data.data) > 1:
                market_data.open = market_data.data[1]
            else:
                market_data.open = getattr(market_data, 'close', 0)
        
        if not hasattr(market_data, 'volume'):
            market_data.volume = 1000
        
        return market_data

    @staticmethod
    def create_compatible_data(price, timestamp, symbol="BTC/USDT"):
        """创建兼容的市场数据"""
        class CompatibleMarketData:
            def __init__(self, price, timestamp, symbol):
                self.symbol = symbol
                self.data = [timestamp, price, price*1.01, price*0.99, price, 1000]
                self.timestamp = timestamp
                self.close = price
                self.open = price
                self.high = price * 1.01
                self.low = price * 0.99
                self.volume = 1000
                self.data_type = 'ohlc'
        
        return CompatibleMarketData(price, timestamp, symbol)