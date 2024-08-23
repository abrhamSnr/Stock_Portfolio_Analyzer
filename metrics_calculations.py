import pandas as pd
import numpy as np
import yfinance as yf 
from datetime import datetime

def calculate_stock_metrics_for_summary_portfolio(stock_metrics_df):
    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    stock_metrics_df_copy = stock_metrics_df.copy()
    inital_investment = 100000
    total_return = stock_metrics_df_copy['total_return'].sum()
    cumulative_return = total_return / inital_investment
    days_hold = ((datetime.strptime(end_date, '%Y-%m-%d')) - (datetime.strptime(start_date, '%Y-%m-%d'))).days
    annualized_return = ((1 + cumulative_return) ** (365/days_hold)) - 1
    volatilty =  stock_metrics_df['volatility'].mean()
    sharp_ratio = calculate_sharp_ratio(annualized_return=annualized_return, volatility=volatilty)
    return {
        'total_return': total_return,
        'cumulative_return': cumulative_return * 100,
        'annulized_return': annualized_return * 100,
        'volatilty': volatilty,
        'sharp_ratio':sharp_ratio
    }

def calculate_metrics_for_each_stock(investment_df):
    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    days_hold = ((datetime.strptime(end_date, '%Y-%m-%d')) - (datetime.strptime(start_date, '%Y-%m-%d'))).days
    investment_df_copy_two = investment_df.copy()
    investment_df_copy_two['daily_return'] = investment_df_copy_two['close'].pct_change()
    total_return = calculate_total_return(investment_df_copy_two)
    volatiltiy = calculate_volatiltiy(investment_df_copy_two)
    investment_df_copy_two['10_days_MA'] = investment_df_copy_two['close'].rolling(window=10).mean()
    investment_df_copy_two['100_days_MA'] = investment_df_copy_two['close'].rolling(window=100).mean()
    cumulative_return = calculate_cumulative_return(investment_df_copy_two)
    annualized_return = ((1 + cumulative_return) ** (365/days_hold)) - 1
    return {
        'total_return': total_return,
        'cumulative_return':  cumulative_return * 100,
        'volatility': volatiltiy * 100,
        '10_days_ma': investment_df_copy_two['10_days_MA'].iloc[-1],
        '100_days_ma': investment_df_copy_two['100_days_MA'].iloc[-1],
        'sharp_ratio': calculate_sharp_ratio(annualized_return, volatiltiy),
    }

def calculate_total_return(investment_df):
    total_return = (investment_df['investment_value'].iloc[-1] - investment_df['investment_value'].iloc[0]) 
    return total_return

def calculate_cumulative_return(investment_df):
    cumulative_return = (investment_df['investment_value'].iloc[-1] - investment_df['investment_value'].iloc[0]) / investment_df['investment_value'].iloc[0]
    return cumulative_return

def calculate_volatiltiy(investment_df):
    volatility = investment_df['daily_return'].std() * np.sqrt(252)
    return volatility

def calculate_sharp_ratio(annualized_return, volatility):
    risk_free_rate = 0.02
    sharp_ratio = (annualized_return - risk_free_rate) / volatility
    return sharp_ratio

def calculate_beta_stock(stock_close_df):
    benchmark_ticker = '^GSPC'
    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)
    benchmark_returns = benchmark_data['Close'].pct_change().fillna(0)
    stock_returns = stock_close_df['close'].pct_change().fillna(0)
    X = np.vstack([benchmark_returns, np.ones(len(benchmark_returns))]).T
    beta, alpha = np.linalg.lstsq(X, stock_returns, rcond=None)[0]
    return {
        'beta': beta
    }