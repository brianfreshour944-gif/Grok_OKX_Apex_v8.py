import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from IPython.display import clear_output

from ml_predictor import SafeMLPredictor, GrokGQA_Transformer 

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(
        self,
        predictor: SafeMLPredictor,
        historical_df: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001, 
        prediction_threshold: float = 0.51, # Adjusted for looser entry parameters
        seq_len: int = 32 # Formally locked to your 8-layer model specifications
    ):
        self.predictor = predictor
        self.historical_df = historical_df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.prediction_threshold = prediction_threshold
        self.seq_len = seq_len

        self.capital = initial_capital
        self.position = 0 
        self.entry_price = 0.0 
        self.equity_curve = []
        self.equity_times = [] 
        self.trades = []
        self.total_wins = 0
        self.total_losses = 0
        self.profits = []
        self.losses = []

        logger.info(f"Backtest Engine Initialized with:\n"\
                    f"  Initial Capital: {initial_capital}\n"\
                    f"  Transaction Cost: {transaction_cost * 100:.2f}%\n"\
                    f"  Prediction Threshold: {prediction_threshold}\n"\
                    f"  Sequence Length: {seq_len}")

    def run_backtest(self):
        logger.info("Starting backtest workflow simulation...")
        
        # 100-bar warmup lookback to allow moving averages/RSI to compute without generating NaNs
        WARMUP_LOOKBACK = 100 
        
        if len(self.historical_df) < WARMUP_LOOKBACK + 1:
            logger.error(f"Not enough historical data for backtest. Requires at least {WARMUP_LOOKBACK + 1} points.")
            return

        # Loop starts at WARMUP_LOOKBACK so sample_df always has historical padding
        for i in range(WARMUP_LOOKBACK, len(self.historical_df)):
            current_time = self.historical_df.index[i]
            
            # Extract historical window chunk for technical feature warmups
            sample_df = self.historical_df.iloc[i - WARMUP_LOOKBACK : i]

            # Execute model inference via MLPredictor wrapper
            prediction = self.predictor.predict(sample_df)

            current_close_price = self.historical_df['close'].iloc[i]
            current_equity_value = self.capital
            if self.position > 0: 
                current_equity_value += self.position * current_close_price 
            
            self.equity_curve.append(current_equity_value)
            self.equity_times.append(current_time) 

            # Long Entry Logic
            if prediction > self.prediction_threshold and self.position == 0: 
                units_to_buy = (self.capital * (1 - self.transaction_cost)) / current_close_price
                self.position = units_to_buy
                self.capital = 0 
                self.entry_price = current_close_price
                self.trades.append({
                    'time': current_time,
                    'type': 'BUY',
                    'price': current_close_price,
                    'units': units_to_buy,
                    'equity': current_equity_value
                })
                logger.info(f"[{current_time}] BUY @ {current_close_price:.2f} (Pred: {prediction:.4f})")
            
            # Position Liquidation Logic
            elif prediction < (1 - self.prediction_threshold) and self.position > 0: 
                trade_profit_loss = (current_close_price - self.entry_price) * self.position - (self.position * current_close_price * self.transaction_cost)
                if trade_profit_loss > 0:
                    self.total_wins += 1
                    self.profits.append(trade_profit_loss)
                else:
                    self.total_losses += 1
                    self.losses.append(trade_profit_loss)

                self.capital = self.position * current_close_price * (1 - self.transaction_cost) 
                self.trades.append({
                    'time': current_time,
                    'type': 'SELL',
                    'price': current_close_price,
                    'units': self.position,
                    'equity': current_equity_value
                })
                self.position = 0 
                self.entry_price = 0.0 
                logger.info(f"[{current_time}] SELL @ {current_close_price:.2f} (Pred: {prediction:.4f})")

        # Handle final trailing open positions safely at termination
        final_equity = self.capital
        if self.position > 0:
            final_equity += self.position * self.historical_df['close'].iloc[-1]
            trade_profit_loss = (self.historical_df['close'].iloc[-1] - self.entry_price) * self.position - (self.position * self.historical_df['close'].iloc[-1] * self.transaction_cost)
            if trade_profit_loss > 0:
                self.total_wins += 1
                self.profits.append(trade_profit_loss)
            else:
                self.total_losses += 1
                self.losses.append(trade_profit_loss)

        if self.equity_curve:
            self.equity_curve[-1] = final_equity 
        else: 
            self.equity_curve.append(final_equity)
            self.equity_times.append(self.historical_df.index[-1])

        logger.info("Backtest simulation successfully compiled.")

    def plot_equity_curve(self):
        if not self.equity_curve:
            logger.warning("No equity curve data available. Please run backtest first.")
            return

        clear_output(wait=True)
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_times, self.equity_curve, label='8-Layer GQA Transformer Equity Line', color='#00ff88', linewidth=2)
        plt.title('Grok GQA Transformer Framework Equity Curve')
        plt.xlabel('Timeline Engine Datetime')
        plt.ylabel('Portfolio Value ($ USD)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

    def calculate_metrics(self):
        if not self.equity_curve:
            logger.warning("No metric analytics available. Please run backtest first.")
            return {}

        equity = pd.Series(self.equity_curve)
        initial = self.initial_capital
        final = equity.iloc[-1]
        total_return = (final - initial) / initial

        returns = equity.pct_change().dropna() 
        if len(returns) > 0 and returns.std() != 0:
            # Assumes 1-minute tracking step execution structure
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) 
        else:
            sharpe_ratio = np.nan

        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        win_rate = self.total_wins / (self.total_wins + self.total_losses) if (self.total_wins + self.total_losses) > 0 else 0
        avg_profit_per_win = np.mean(self.profits) if self.profits else 0
        avg_loss_per_loss = np.mean(self.losses) if self.losses else 0

        metrics = {
            'Initial Capital': initial,
            'Final Capital': final,
            'Total Return': f'{total_return:.2%}',
            'Sharpe Ratio (Annualized)': f'{sharpe_ratio:.2f}' if not np.isnan(sharpe_ratio) else 'N/A',
            'Max Drawdown': f'{max_drawdown:.2%}',
            'Number of Trades': len([t for t in self.trades if t['type'] == 'BUY']),
            'Total Wins': self.total_wins,
            'Total Losses': self.total_losses,
            'Win Rate': f'{win_rate:.2%}',
            'Average Profit per Win': f'${avg_profit_per_win:.2f}',
            'Average Loss per Loss': f'${avg_loss_per_loss:.2f}'
        }

        print("\n--- Backtest Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("------------------------")

        return metrics
