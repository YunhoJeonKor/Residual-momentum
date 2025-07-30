from vbase_utils.stats.pit_robust_betas import pit_robust_betas, backtest
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
# import pandas_market_calendars as mcal
import logging
def produce_portfolio(portfolio_date: str, logger: object) -> pd.DataFrame:
    """
    Produces an ETF portfolio based on residuals from a factor model using pit_robust_betas.

    The function downloads historical price data for a wide range of ETFs, computes returns,
    estimates betas and residuals using a market factor (IWV), and constructs a sector-neutral
    portfolio based on signals from the residuals using a backtest procedure.

    Args:
        portfolio_date (str): The date for which to construct the portfolio (format: "YYYY-MM-DD").
        logger (object): A logger instance for logging steps and diagnostics.

    Returns:
        pd.DataFrame: A DataFrame containing portfolio weights with columns ['sym', 'wt'].

    Raises:
        ValueError: If the data download fails or contains no price information.
    """
    tickers = ['IWV','RPV','VOE','VBR','IWD','IWS','IWN','IWX','IUSV','DTH','FDL','EFV','FTA','DON','PID','PFM','PXF','PWV','RPG','VUG','VBK','IVW','IWO','IWP','IUSG','SCHG','SPYG','EPS','FAD','FTC',
    'FTCS','VYM','SDY','DHS','DVY','PEY','FVD','DEW','IJS','PRFZ','RZV','RZG','FYX','DFE','DLS','DWX','VOT','MDYV','MDYG','IJK','IJT','FNX','RWK','RFG','RFV','SPHQ','IWF','DLN','DTD','FEX']
    etf_assortment = [['RPV','VOE','VBR','IWD','IWS','IWN','IWX','IUSV','DTH','EFV','FDL','FTA','DON','PID','PFM','PXF','PWV'],
    ['RPG','VUG','VBK','IVW','IWO','IWP','IUSG','SCHG','SPYG','EPS','FAD','FTC','FTCS'],['VYM', 'SDY', 'DHS', 'DVY', 'PEY', 'FVD', 'DEW'],['IJS', 'PRFZ', 'RZV', 'RZG', 'FYX', 'DFE', 'DLS', 'DWX'],
    ['VOT', 'MDYV', 'MDYG', 'IJK', 'IJT', 'FNX', 'RWK', 'RFG', 'RFV'],['SPHQ', 'IWF', 'DLN', 'DTD', 'FEX']]

    portfolio_datetime = datetime.strptime(portfolio_date, "%Y-%m-%d")
    current_datetime = datetime.now()
    is_current_date = portfolio_datetime.date() == current_datetime.date()
    logger.info(f"Current time: {current_datetime.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Is current date: {is_current_date}")
    end_date = portfolio_datetime
    start_date = end_date - timedelta(days=150)

    logger.info(f"Fetching price data from {start_date.date()} to {end_date.date()}...")


    df = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if df.empty:
        logger.error("Downloaded price data is empty.")
        raise ValueError("No price data retrieved from Yahoo Finance.")

    returns = df['Close'].pct_change()
    df_rets = returns.iloc[1:]
    df_rets.index = pd.to_datetime(df_rets.index, utc=True).tz_localize(None)
    weekly_rebalance = pd.DatetimeIndex([dt for dt in df_rets.index if dt.weekday() == 4])

    if weekly_rebalance.empty:
        logger.warning("No Fridays found in return data for weekly rebalance.")

    market_returns = df_rets[["IWV"]]
    asset_returns = df_rets.drop("IWV", axis=1)

    logger.info("Calculating pit_robust_betas...")


    market_results = pit_robust_betas(
        df_asset_rets=asset_returns,
        df_fact_rets=market_returns,
        # Approximately 6 months.
        half_life=126,
        # Approximately 3 months.
        min_timestamps=63,
        rebalance_time_index=weekly_rebalance,
        progress=True,
    )
    market_residuals = market_results["df_asset_resids"]
    
    logger.info("Running backtest with residuals...")
    
    position_df =  backtest(df_rets, market_residuals,market_returns, etf_assortment,
        signal_window = 1, vol_window = 20, vol_scale = False, plot_true = True, sector_neutral= True, exponential_weights= True, percentile = False, market_hedged = False,
            beta_window= 40, beta_hedge=False, rolling_window = False, rolling_window_size=4).iloc[-1].reset_index()
    position_df.columns = ["sym",'wt']

    logger.info(f"Generated portfolio with HEAD 5:\n{position_df.head(5)}")
    return position_df


# logger = logging.getLogger("portfolio_logger")
# logger.setLevel(logging.INFO)
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#     logger.addHandler(handler)

# today_str = date.today().strftime("%Y-%m-%d")

# produce_portfolio(today_str, logger)