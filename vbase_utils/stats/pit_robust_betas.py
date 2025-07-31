"""Point-in-time robust regression module for calculating hedge ratios and residuals."""
import numpy as np
import logging
from typing import Dict, Optional

import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..'))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# print(os.getcwd())
# print(os.path.dirname(__file__))
from vbase_utils.sim import sim
from vbase_utils.stats.robust_betas import robust_betas

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# The function must take a large number of arguments
# and consequently has a large number of local variables.
# pylint: disable=too-many-arguments, too-many-locals
from typing import Optional,Dict
def pit_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None,
    min_timestamps: int = 10,
    rebalance_time_index: Optional[pd.DatetimeIndex] = None,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Calculate point-in-time robust betas and residuals for time series regressions.

    This function:
    1. Validates and aligns input data
    2. Uses sim() to run robust_betas() at each timestamp
    3. Calculates residuals for t+1 using betas known at t
    4. Returns both the betas and residuals as DataFrames

    Args:
        df_asset_rets: DataFrame of dependent returns with shape (n_timestamps, n_assets).
        df_fact_rets: DataFrame of factor returns with shape (n_timestamps, n_factors).
        half_life: Half-life in time units (e.g., days). Must be positive.
        lambda_: Decay factor (e.g., 0.985). Must be between 0 and 1.
        min_timestamps: Minimum number of timestamps required for regression. Defaults to 10.
        rebalance_time_index: Optional DatetimeIndex specifying when to rebalance hedge ratios.
            If not provided, uses all timestamps from df_asset_rets.
        progress: Whether to show a progress bar during simulation. Defaults to False.

    Returns:
        Dictionary containing:
        - 'df_betas': DataFrame of shape (n_timestamps, n_factors, n_assets) containing
          the computed betas at each timestamp
        - 'df_hedge_rets_by_fact': DataFrame of shape (n_timestamps, n_factors, n_assets) containing
          the hedge returns by factor at each timestamp
        - 'df_hedge_rets': DataFrame of shape (n_timestamps, n_assets) containing
          the hedge returns at each timestamp
        - 'df_asset_resids': DataFrame of shape (n_timestamps, n_assets) containing
          the asset residuals at each timestamp

    Raises:
        ValueError: If inputs are empty, have insufficient data, mismatched rows,
            or if timestamps don't align.
    """
    # Validate input data
    if df_asset_rets.empty or df_fact_rets.empty:
        raise ValueError("Input DataFrames cannot be empty")
    # Ensure indices are DatetimeIndex
    if not isinstance(df_asset_rets.index, pd.DatetimeIndex):
        raise ValueError("df_asset_rets must have a DatetimeIndex")
    if not isinstance(df_fact_rets.index, pd.DatetimeIndex):
        raise ValueError("df_fact_rets must have a DatetimeIndex")
    # Ensure timestamps are sorted.
    if not df_asset_rets.index.is_monotonic_increasing:
        df_asset_rets.sort_index(inplace=True)
    if not df_fact_rets.index.is_monotonic_increasing:
        df_fact_rets.sort_index(inplace=True)
    # Ensure the indices are the same.
    if not df_asset_rets.index.equals(df_fact_rets.index):
        raise ValueError("df_asset_rets and df_fact_rets must have the same index")

    # If rebalance_time_index is not provided, use the asset returns index.
    if rebalance_time_index is None:
        rebalance_time_index = df_asset_rets.index

    # Define callback function for sim
    def regression_callback(
        data: Dict[str, pd.DataFrame | pd.Series],
    ) -> Dict[str, pd.DataFrame | pd.Series]:
        """Callback function to run robust regression on masked data."""
        df_asset_rets = data["df_asset_rets"]
        df_fact_rets = data["df_fact_rets"]

        # Run robust regression
        beta_matrix = robust_betas(
            df_asset_rets,
            df_fact_rets,
            half_life=half_life,
            lambda_=lambda_,
            min_timestamps=min_timestamps,
        )

        dict_ret = {
            "betas": beta_matrix,
        }
        return dict_ret

    # Create NA Series for each asset's betas.
    # We will update this with the actual values from the simulation.
    asset_names = df_asset_rets.columns
    factor_names = list(df_fact_rets.columns)
    results = {
        "betas": pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [rebalance_time_index, factor_names], names=["timestamp", "factor"]
            ),
            columns=asset_names,
            dtype=float,
        )
    }

    # Run simulation only for timestamps after min_timestamps.
    if len(df_asset_rets.index) > min_timestamps:
        sim_results = sim(
            {"df_asset_rets": df_asset_rets, "df_fact_rets": df_fact_rets},
            regression_callback,
            rebalance_time_index,
            progress=progress,
        )
        # Fill in the betas DataFrame with the actual values from the simulation.
        results["betas"].update(sim_results["betas"])

    # Calculate residuals using matrix operations.

    # Get the betas DataFrame.
    df_betas = results["betas"]

    # Reindex betas to the new MultiIndex and fill in missing values
    # Create a MultiIndex for the asset returns index
    new_index = pd.MultiIndex.from_product(
        [df_asset_rets.index, factor_names], names=["timestamp", "factor"]
    )
    df_betas = df_betas.reindex(new_index)

    # Forward fill betas along the timestamp index to match return timestamps.
    df_betas.ffill(inplace=True, axis=0)

    # Shift the betas by 1 day.
    # This ensures we use betas from t-1 to hedge returns at t
    # and gives us the effective hedge weights at t.
    df_hedge_weights = -1 * df_betas.shift(1)

    # Calculate the predicted returns.
    # We must unstack the factor name column to an index level.
    # Transform to MultiIndex format.
    df_fact_rets_stacked = df_fact_rets.stack().to_frame()
    df_fact_rets_stacked.index.names = ["timestamp", "factor"]
    df_fact_rets_stacked.columns = ["ret"]
    # Multiply the hedge weights by the factor returns for each factor
    # Using multiplication with align.
    df_hedge_rets_by_fact = df_hedge_weights.multiply(
        df_fact_rets_stacked["ret"], axis=0
    )
    # Sum across factors for each timestamp-asset combination, then unstack.
    df_hedge_rets = df_hedge_rets_by_fact.groupby("timestamp").sum(min_count=1)

    # Calculate the resids.
    df_asset_resids = df_asset_rets + df_hedge_rets
    # Set the index names.
    # df_asset_rets may not have the index name specified.
    if df_asset_resids.index.name is None:
        df_asset_resids.index.name = "timestamp"

    return {
        "df_betas": df_betas,
        "df_hedge_rets_by_fact": df_hedge_rets_by_fact,
        "df_hedge_rets": df_hedge_rets,
        "df_asset_resids": df_asset_resids,
    }

def compute_rolling_betas(df_rets, market_series, window):

    cov = df_rets.rolling(window).cov(market_series)
    
    var = market_series.rolling(window).var()

    betas = cov.div(var, axis=0)
    return betas

def project_IP_form_cash0_beta1(
    weights: pd.DataFrame,
    beta: pd.DataFrame,
    target_beta: float = 1.0,
    eps: float = 1e-12,
    fallback: str = "cash_only",  # {"cash_only","nan","keep"}
) -> tuple[pd.DataFrame, list]:
    
    W, B = weights.align(beta, axis=1, join="inner")
    W, B = W.align(B, axis=0, join="inner")

    out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    bad_dates = []

    for t, w in W.iterrows():
        b = B.loc[t]
        mask = w.notna() & b.notna()
        k = int(mask.sum())
        if k == 0:
            continue

        wv = w[mask].to_numpy(float)     # (k,)
        bv = b[mask].to_numpy(float)     # (k,)
        ones = np.ones(k, dtype=float)

        # --- P = U (U^T U)^{-1} U^T,  U=[1, beta]
        UtU = np.array([[k,         bv.sum()],
                        [bv.sum(),  np.dot(bv, bv)]], dtype=float)
        det = UtU[0,0]*UtU[1,1] - UtU[0,1]*UtU[1,0]
        if abs(det) < eps:
            bad_dates.append(t)
            if fallback == "cash_only":
                out.loc[t, mask] = wv - wv.mean()   
            elif fallback == "nan":
                out.loc[t, mask] = np.nan
            elif fallback == "keep":
                out.loc[t, mask] = wv
            continue

        invUtU = (1.0/det) * np.array([[ UtU[1,1], -UtU[0,1]],
                                       [-UtU[1,0],  UtU[0,0]]], dtype=float)
        U      = np.column_stack([ones, bv])       
        Pwv    = U @ (invUtU @ (U.T @ wv))         
        w_orth = wv - Pwv                          

        c = np.array([0.0, float(target_beta)])
        coeff = invUtU @ c             
        u_part = U @ coeff             

        w_star = w_orth + u_part
        out.loc[t, mask] = w_star

    return out, bad_dates


def neutralize_cash_beta_df(
    weights: pd.DataFrame,
    beta: pd.DataFrame,
    keep_l1_norm: bool = True,
    cond_thresh: float = 1e12,
    cash_only_when_singular: bool = True,
) -> tuple[pd.DataFrame, list]:
    
    W, B = weights.align(beta, axis=1, join="inner")
    W, B = W.align(B, axis=0, join="inner")

    out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    singular_dates = []

    for t, w in W.iterrows():
        b = B.loc[t]
        mask = w.notna() & b.notna()
        k = int(mask.sum())
        if k < 2:
            continue

        wv = w[mask].to_numpy(float)
        bv = b[mask].to_numpy(float)
        U = np.column_stack([np.ones(k), bv])  # [1, beta_t] (k×2)

        M = U.T @ U
        try:

            if np.linalg.cond(M) > cond_thresh:
                raise np.linalg.LinAlgError("ill-conditioned")

            a = np.linalg.solve(M, U.T @ wv)   
            wn = wv - U @ a                    
        except np.linalg.LinAlgError:
            singular_dates.append(t)
            if cash_only_when_singular:
                wn = wv - wv.mean()          
            else:
                continue

        if keep_l1_norm:
            s_raw = np.abs(wv).sum()
            s_neu = np.abs(wn).sum()
            if s_neu > 0:
                wn *= (s_raw / s_neu)

        out.loc[t, mask] = wn

    return out, singular_dates



def backtest(df_rets, market_residuals, market_returns, etf_assortment, rf_daily = None, signal_window = 1, vol_window = 20, vol_scale = False, plot_true = True,
              exponential_weights = True, sector_neutral = True, percentile = True, market_hedged = False, beta_window = 60, weights_clip = False, max_weight = 0.05
              , beta_hedge = False, lag = 1, rolling_window = True, rolling_window_size = 3, corr= True, market_1 = True, beta_neutralize = False, w_nue = False, beta_0 = True):
    
    sector_resid = market_residuals
    if vol_scale:
        vol = sector_resid.rolling(vol_window).std().dropna()
        scaled_sector_resid = sector_resid / (vol + 1e-6)
        cumulative_resid = (1+scaled_sector_resid).rolling(window=signal_window).apply(np.prod, raw=True) - 1 
        daily_reverse_signal = - cumulative_resid
    else:
        cumulative_resid = (1+sector_resid).rolling(window=signal_window).apply(np.prod, raw=True) - 1 
        daily_reverse_signal = - cumulative_resid
    
    z_scores_group = daily_reverse_signal.sub(cumulative_resid.mean(axis=1), axis = 0).div(cumulative_resid.std(axis=1), axis = 0)
    if sector_neutral:
        for etfs in etf_assortment:
            z = z_scores_group[etfs]
            z_sector_adj = z.sub(z.mean(axis=1), axis=0)
            z_scores_group[etfs] = z_sector_adj
        z_all = z_scores_group
    else:
        z_all = z_scores_group

    z_scores = (z_all.sub(z_all.mean(axis=1), axis  = 0).div(z_all.std(axis=1), axis = 0))
    long_signals = z_scores.clip(lower=0)
    short_signals = z_scores.clip(upper=0)
    long_signals = long_signals.div(long_signals.sum(axis=1), axis=0).fillna(0) * 1/5
    short_signals = short_signals.div(short_signals.abs().sum(axis=1), axis=0).fillna(0) * 1/5
    z_scores = long_signals + short_signals

    betas = compute_rolling_betas(df_rets, market_returns['IWV'], 60)
    betas.dropna()
    if beta_0:
        w_proj, bad = neutralize_cash_beta_df(z_scores, betas)
        weights = w_proj
    else:
        w_proj, bad = project_IP_form_cash0_beta1(z_scores,betas)
        weights = w_proj
    print("This is attached in only this file")
    return weights

# def backtest(df_rets, market_residuals, market_returns, etf_assortment, rf_daily = None, signal_window = 1, vol_window = 20, vol_scale = False, plot_true = True,
#               exponential_weights = True, sector_neutral = True, percentile = True, market_hedged = False, beta_window = 60, weights_clip = False, max_weight = 0.05
#               , beta_hedge = False, lag = 1, rolling_window = True, rolling_window_size = 3, corr= True, market_1 = True):
    
#     sector_resid = market_residuals
#     if vol_scale:
#         vol = sector_resid.rolling(vol_window).std().dropna()
#         scaled_sector_resid = sector_resid / (vol + 1e-6)
#         cumulative_resid = (1+scaled_sector_resid).rolling(window=signal_window).apply(np.prod, raw=True) - 1 
#         daily_reverse_signal = - cumulative_resid
#     else:
#         cumulative_resid = (1+sector_resid).rolling(window=signal_window).apply(np.prod, raw=True) - 1 
#         daily_reverse_signal = - cumulative_resid
    
#     z_scores_group = daily_reverse_signal.sub(cumulative_resid.mean(axis=1), axis = 0).div(cumulative_resid.std(axis=1), axis = 0)
#     if sector_neutral:
#         for etfs in etf_assortment:
#             z = z_scores_group[etfs]
#             z_sector_adj = z.sub(z.mean(axis=1), axis=0)
#             z_scores_group[etfs] = z_sector_adj
#         z_all = z_scores_group
#     else:
#         z_all = z_scores_group
#     if not percentile:
#         z_scores = (z_all.sub(z_all.mean(axis=1), axis  = 0).div(z_all.std(axis=1), axis = 0))
#         long_signals = z_scores.clip(lower=0)
#         short_signals = z_scores.clip(upper=0)
        
#         long_weights = long_signals.div(long_signals.sum(axis=1), axis=0).fillna(0) * 1/2
#         short_weights = short_signals.div(short_signals.abs().sum(axis=1), axis=0).fillna(0) * 1/2
#         weights = long_weights + short_weights
#         if exponential_weights:
#             weights = weights.ewm(span=10, adjust=False).mean()

#             long_weights = weights.clip(lower=0)
#             short_weights = weights.clip(upper=0)

#             long_weights = long_weights.div(long_weights.sum(axis=1), axis=0).fillna(0) * 1/2
#             short_weights = short_weights.div(short_weights.abs().sum(axis=1), axis=0).fillna(0) * 1/2

#             weights = long_weights + short_weights
#     else:
#         z_scores = (z_all.sub(z_all.mean(axis=1), axis=0)
#                   .div(z_all.std(axis=1), axis=0))

#         quantile = 0.1
#         top_mask = z_scores.ge(z_scores.quantile(1 - quantile, axis=1), axis=0)
#         bottom_mask = z_scores.le(z_scores.quantile(quantile, axis=1), axis=0)

#         long_signals = z_scores.where(top_mask, 0)
#         short_signals = z_scores.where(bottom_mask, 0)
#         long_weights = long_signals.div(long_signals.sum(axis=1), axis=0).fillna(0) * 1/2
#         short_weights = short_signals.div(short_signals.abs().sum(axis=1), axis=0).fillna(0) * 1/2

#         weights = long_weights + short_weights

#     return weights