import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import clean_opt_chain, implied_vol, compute_risk_neutral_pdf

# -- Parameters
date_strs = ['08-04-2025', '09-04-2025']
maturity = datetime(2025, 5, 1)
# set a range of strikes with very dense data, in order to deal with data sparsity
lower_strike = 5000 
upper_strike = 6000
skip_strike_parameter = 15

# -- Set up for single plot
sns.set_theme(style="whitegrid", palette="Set2")
plt.figure(figsize=(12, 7))
colors = sns.color_palette("Set2", n_colors=len(date_strs))

# -- Runner
for idx, date_label in enumerate(date_strs):

    # 1) Load data
    prices = pd.read_csv('SP500_data.csv')
    calls = pd.read_excel('SP500_chain_data.xlsx', sheet_name=date_label)

    # 2) Extract parameters
    S = prices.loc[prices['Date'] == date_label, 'Close'].values[0]
    date = datetime.strptime(date_label, '%d-%m-%Y')
    t = (maturity - date).days / 365
    
    # 3) Preprocess option chain
    calls = clean_opt_chain(calls, lower_strike, upper_strike, skip_strike_parameter)
    calls["iv"] = calls.apply(lambda row: implied_vol(row.Midprice, S, row.Strike, t), axis=1)

    # 4) Compute risk-neutral PDF
    x_vals, pdf = compute_risk_neutral_pdf(calls, S, t)

    # 5) Plot
    plt.plot(x_vals, pdf, lw=2.5, label=f"{date_label} (S={S:.2f})", color=colors[idx])
    plt.axvline(S, color=colors[idx], linestyle=':', lw=2) # plot vertical line for S
    plt.text(S, max(pdf)*0.95, f"S={S:.0f}", rotation=90,
             verticalalignment='top', horizontalalignment='right',
             fontsize=10, color=colors[idx], fontweight='bold')

# -- Finalize the plot
plt.xlabel('Strike Price (K)', fontsize=14)
plt.ylabel('Risk-Neutral PDF f(K)', fontsize=14)
plt.title("SP500 Risk-Neutral Densities", fontsize=16, fontweight='bold', pad=20)
plt.legend(title="Date", fontsize=11)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine()
plt.tight_layout()
plt.show()
