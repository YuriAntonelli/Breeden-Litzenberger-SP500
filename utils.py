import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from scipy.optimize import minimize_scalar

def clean_opt_chain(df, lower_bound, upper_bound, n_skips):
    # 1) take the most dense part of the option chain
    dense_df = df[(df['Strike'] >= lower_bound) & (df['Strike'] <= upper_bound)]
    # 2) keep less granular data
    sampled_df = dense_df.iloc[::n_skips]
    # 3) rebuild a full chain 
    outside_df = df[(df['Strike'] < lower_bound) | (df['Strike'] > upper_bound)]
    filtered_calls = pd.concat([sampled_df, outside_df], ignore_index=True)
    # 4) add midprice
    filtered_calls["Midprice"] = (filtered_calls.Bid + filtered_calls.Ask)/2
    filtered_calls = filtered_calls[filtered_calls.Midprice > 0]
    return filtered_calls


def call_value(S, K, sigma, t=0, r=0.02):
    # use np.multiply and divide to handle divide-by-zero
    with np.errstate(divide='ignore'):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S/K) + (r+sigma**2 / 2) * t)
        d2 = d1 - sigma * np.sqrt(t)
    return np.multiply(norm.cdf(d1),S) - np.multiply(norm.cdf(d2), K * np.exp(-r * t))


def implied_vol(opt_value, S, K, T, r=0.02):
    def call_obj(sigma):
        return abs(call_value(S, K, sigma,T, r) - opt_value)
    res = minimize_scalar(call_obj, bounds=(0.01,6), method='bounded')
    return res.x


def compute_risk_neutral_pdf(df, S, t, r=0.02):
    df = df.sort_values('Strike')

    # Interpolate implied vol surface
    vol_surface = scipy.interpolate.interp1d(
        df.Strike, df.iv, kind="cubic", fill_value="extrapolate"
    )

    # Define strike range
    x_new = np.arange(df.Strike.min(), df.Strike.max(), 1)

    # Compute call prices
    C_vals = call_value(S, x_new, vol_surface(x_new), t, r)

    # Numerical differentiation
    first_deriv = np.gradient(C_vals, x_new, edge_order=0)
    second_deriv = np.gradient(first_deriv, x_new, edge_order=0)

    # Risk-neutral PDF
    pdf_vals = np.exp(r * t) * second_deriv

    return x_new, pdf_vals