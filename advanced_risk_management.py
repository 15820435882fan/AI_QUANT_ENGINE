# advanced_risk_management.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class AdvancedRiskManager:
    """高级风险管理系统"""
    
    def __init__(self):
        self.risk_metrics = {}
        self.alert_history = []
        self.setup_risk_parameters()
    
    def setup_risk_parameters(self):
        """设置风险参数"""
        self.risk_limits = {
            'max_drawdown': 0.15,        # 最大回撤15%
            'daily_loss_limit': 0.05,    # 单日最大损失5%
            'position_limit': 0.2,       # 单仓位最大20%
            'sector_exposure': 0.5,      # 单一板块最大50%
            'var_confidence': 0.95,      # VaR置信度95%
        }
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        if len(returns) < 30:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var)
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> Dict[str, float]:
        """计算最大回撤"""
        if len(portfolio_values) < 2:
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0}
        
        peak = portfolio_values[0]
        max_dd = 0.0
        current_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
            current_dd = drawdown
        
        return {
            'max_drawdown': max_dd,
            'current_drawdown': current_dd
        }
    
    def analyze_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析投资组合风险"""
        risk_report = {}
        
        try:
            # 计算最大回撤
            if 'portfolio_values' in portfolio_data:
                drawdown_analysis = self.calculate_max_drawdown(
                    portfolio_data['portfolio_values']
                )
                risk_report['drawdown'] = drawdown_analysis
            
            # 计算波动率
            if 'returns' in portfolio_data and len(portfolio_data['returns']) > 1:
                returns = pd.Series(portfolio_data['returns'])
                risk_report['volatility'] = returns.std()
                risk_report['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # 计算VaR
            if 'returns' in portfolio_data and len(portfolio_data['returns']) > 30:
                var = self.calculate_var(returns, self.risk_limits['var_confidence'])
                risk_report['var_95'] = var
            
            # 检查风险限额
            risk_report['limit_checks'] = self.check_risk_limits(risk_report, portfolio_data)
            
            # 总体风险评估
            risk_report['overall_risk'] = self.assess_overall_risk(risk_report)
            
        except Exception as e:
            risk_report['error'] = f"风险分析失败: {e}"
        
        return risk_report
    
    def check_risk_limits(self, risk_metrics: Dict, portfolio_data: Dict) -> Dict[str, bool]:
        """检查风险限额"""
        checks = {}
        
        # 检查最大回撤
        if 'drawdown' in risk_metrics:
            current_dd = risk_metrics['drawdown']['current_drawdown']
            checks['drawdown_within_limit'] = current_dd <= self.risk_limits['max_drawdown']
        
        # 检查仓位集中度
        if 'positions' in portfolio_data:
            total_value = portfolio_data.get('total_value', 1)
            for symbol, position in portfolio_data['positions'].items():
                position_pct = position['value'] / total_value
                checks[f'position_{symbol}'] = position_pct <= self.risk_limits['position_limit']
        
        return checks
    
    def assess_overall_risk(self, risk_metrics: Dict) -> str:
        """评估总体风险水平"""
        risk_score = 0
        
        if 'drawdown' in risk_metrics:
            dd = risk_metrics['drawdown']['current_drawdown']
            if dd > 0.1:
                risk_score += 2
            elif dd > 0.05:
                risk_score += 1
        
        if 'volatility' in risk_metrics:
            vol = risk_metrics['volatility']
            if vol > 0.03:
                risk_score += 1
        
        if risk_score >= 2:
            return "HIGH"
        elif risk_score == 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def should_allow_trade(self, trade_data: Dict, portfolio_data: Dict) -> Dict[str, Any]:
        """判断是否允许交易"""
        risk_analysis = self.analyze_portfolio_risk(portfolio_data)
        
        decision = {
            'allowed': True,
            'reason': '风险检查通过',
            'risk_level': risk_analysis.get('overall_risk', 'LOW'),
            'checks': risk_analysis.get('limit_checks', {})
        }
        
        # 检查风险限额
        if not all(risk_analysis.get('limit_checks', {}).values()):
            decision['allowed'] = False
            decision['reason'] = '超过风险限额'
        
        # 检查总体风险水平
        if risk_analysis.get('overall_risk') == 'HIGH':
            decision['allowed'] = False
            decision['reason'] = '总体风险水平过高'
        
        # 记录决策
        self.alert_history.append({
            'timestamp': datetime.now(),
            'trade_data': trade_data,
            'decision': decision,
            'risk_analysis': risk_analysis
        })
        
        return decision

def test_risk_management():
    """测试风险管理系统"""
    print("🧪 测试高级风险管理系统...")
    
    risk_manager = AdvancedRiskManager()
    
    # 模拟投资组合数据
    portfolio_data = {
        'portfolio_values': [10000, 10500, 10200, 9800, 10100, 9900],
        'returns': [0.05, -0.028, -0.039, 0.031, -0.019],
        'total_value': 9900,
        'positions': {
            'BTC-USDT': {'value': 2000},
            'ETH-USDT': {'value': 1500}
        }
    }
    
    # 分析风险
    risk_report = risk_manager.analyze_portfolio_risk(portfolio_data)
    print(f"📊 风险报告: {risk_report}")
    
    # 测试交易审批
    trade_data = {
        'symbol': 'BTC-USDT',
        'action': 'BUY',
        'size': 1000
    }
    
    decision = risk_manager.should_allow_trade(trade_data, portfolio_data)
    print(f"🎯 交易决策: {decision}")
    
    return risk_manager

if __name__ == "__main__":
    test_risk_management()