import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("csv_files/weight-height.csv")
print(df.head())
print(df.describe())

upper_limit = df["Height"].quantile(0.99)
lower_limit = df["Height"].quantile(0.01)

print(df[(df["Height"]>=upper_limit) | (df["Height"]<=lower_limit)])

# # ===trimming===

new_df = df[(df["Height"]<=upper_limit) | (df["Height"]>=lower_limit)]
print(new_df.head())

# # ===capping===

new_df = df.copy()

new_df = np.where(new_df["Height"]>upper_limit,upper_limit,np.where(new_df["Height"]<lower_limit,lower_limit,new_df["Height"]))

sns.distplot(df["Height"])
plt.show()