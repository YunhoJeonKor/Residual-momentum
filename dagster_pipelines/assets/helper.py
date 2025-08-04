import numpy as np
import logging
from typing import Dict, Optional

import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def compute_rolling_betas(df_rets, market_series, window):

    cov = df_rets.rolling(window).cov(market_series)
    
    var = market_series.rolling(window).var()

    betas = cov.div(var, axis=0)
    return betas

def project_IP_form_cash0_beta1(
    weights,
    beta,
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

        c = np.array([1.0, float(target_beta)])
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
    long_signals = long_signals.div(long_signals.sum(axis=1), axis=0).fillna(0)
    short_signals = short_signals.div(short_signals.abs().sum(axis=1), axis=0).fillna(0) 
    z_scores = long_signals + short_signals
    long_weights = long_signals.div(long_signals.sum(axis=1), axis=0).fillna(0) * 1/2
    short_weights = short_signals.div(short_signals.abs().sum(axis=1), axis=0).fillna(0) * 1/2
    if not beta_0:
        long_weights = long_signals.div(long_signals.sum(axis=1), axis=0).fillna(0) * 1.5
        short_weights = short_signals.div(short_signals.abs().sum(axis=1), axis=0).fillna(0) * 1/2
    weights = long_weights + short_weights
    if exponential_weights:
        weights = weights.ewm(span=10, adjust=False).mean()

        long_weights = weights.clip(lower=0)
        short_weights = weights.clip(upper=0)
        if beta_0:
            long_weights = long_weights.div(long_weights.sum(axis=1), axis=0).fillna(0) * 1/2
            short_weights = short_weights.div(short_weights.abs().sum(axis=1), axis=0).fillna(0) * 1/2
        elif not beta_0:
            long_weights = long_weights.div(long_weights.sum(axis=1), axis=0).fillna(0) * 1.5
            short_weights = short_weights.div(short_weights.abs().sum(axis=1), axis=0).fillna(0) * 1/2

        weights = long_weights + short_weights

    betas = compute_rolling_betas(df_rets, market_returns['IWV'], 60)
    if beta_0:
        w_proj, bad = neutralize_cash_beta_df(weights, betas)
        weights = w_proj
            
        long_weights = weights.clip(lower=0)
        short_weights = weights.clip(upper=0)

        long_weights = long_weights.div(long_weights.sum(axis=1), axis=0).fillna(0) * 1/2
        short_weights = short_weights.div(short_weights.abs().sum(axis=1), axis=0).fillna(0) * 1/2

        weights = long_weights + short_weights
    else:
        w_proj, bad = project_IP_form_cash0_beta1(weights,betas)
        weights = w_proj
    
        long_weights = weights.clip(lower=0)
        short_weights = weights.clip(upper=0)

        long_weights = long_weights.div(long_weights.sum(axis=1), axis=0).fillna(0) * 1.1
        short_weights = short_weights.div(short_weights.abs().sum(axis=1), axis=0).fillna(0) * 1/10

        weights = long_weights + short_weights
    return weights