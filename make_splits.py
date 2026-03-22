import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train_users.csv", index_col="id_user")
df["timestamp_reg"] = pd.to_datetime(df["timestamp_reg"], format="mixed")
df = df.sample(frac=1)
df["split"] = ((np.arange(len(df)) / len(df)) > 0.8).astype(int)

df["split"].to_csv("dataset/splits.csv")
