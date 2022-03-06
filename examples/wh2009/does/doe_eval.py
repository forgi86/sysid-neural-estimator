import os
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import pandas as pd


if __name__ == '__main__':

    sns.set_theme(style="whitegrid")

    DOE_NAME = "doe1"

    df_res = pd.read_csv(DOE_NAME + "_res.csv")  # doe1_res.csv
    df_res.sort_values(by="FIT", inplace=True, ascending=False)

    factors = ["max_time", "batch_size", "seq_len", "est_frac", "est_direction", "est_type", "est_hidden_size"]
    response = "FIT"
    for factor in factors:
        df_res[factor] = df_res[factor].astype("category")
    g = sns.PairGrid(df_res, y_vars=response,
                     x_vars=factors,
                     height=5, aspect=.5)

    # Draw a seaborn point plot onto each Axes
    g.map(sns.pointplot, scale=1.3)
    sns.despine(fig=g.fig, left=True)

    print(df_res.head(10))
