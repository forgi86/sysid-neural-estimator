import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == '__main__':

    sns.set_theme(style="whitegrid")

    DOE_NAME = "doe1"

    df_res = pd.read_csv(DOE_NAME + "_res.csv")  # doe1_res.csv
    df_res.sort_values(by="FIT", inplace=True, ascending=False)
    df_res["FIT"] = df_res["FIT"].fillna(0.0)
    df_res["FIT"] = df_res["FIT"] * (df_res["FIT"] > 0)
    df_res["RMSE"] = df_res["RMSE"].fillna(1000)

    factors = ["max_time", "seq_len", "est_frac", "est_direction", "est_type", "est_hidden_size"]
    response = "FIT"
    for factor in factors:
        df_res[factor] = df_res[factor].astype("category")
    g = sns.PairGrid(df_res, y_vars=response,
                     x_vars=factors,
                     height=5, aspect=.5)

    #%% Draw a seaborn point plot onto each Axes
    g.map(sns.pointplot, scale=1.3)
    sns.despine(fig=g.fig, left=True)

    # df_good = df_res[(df_res["seq_len"] != 40) & (df_res["est_direction"] == "forward")]
    df_good = df_res[(df_res["seq_len"] != 40)]# & (df_res["est_direction"] == "forward")]
    factors = ["max_time", "seq_len", "est_frac", "est_direction", "est_type", "est_hidden_size"]
    response = "FIT"
    for factor in factors:
        df_res[factor] = df_res[factor].astype("category")

    g = sns.PairGrid(df_good, y_vars=response,
                     x_vars=factors,
                     height=5, aspect=.5)

    # Draw a seaborn point plot onto each Axes
    g.map(sns.pointplot, scale=1.3)
    sns.despine(fig=g.fig, left=True)

    #%%
    g = sns.PairGrid(df_good, y_vars=response,
                     x_vars=factors,
                     height=5, aspect=.5)
    # Draw a seaborn point plot onto each Axes
    g.map(sns.histplot)#, scale=1.3)
    sns.despine(fig=g.fig, left=True)

    #%%
    print(df_res.head(10))
