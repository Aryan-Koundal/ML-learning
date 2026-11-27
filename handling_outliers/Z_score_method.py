import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("csv_files/placement.csv")
print(df.head())


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.displot(df["cgpa"],kde=True,alpha=0.4,edgecolor="none")

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

plt.show()


# # ===trimming===


heighest_val = df["cgpa"].mean()+3*df["cgpa"].std()
lowest_val = df["cgpa"].mean()-3*df["cgpa"].std()

# print(df[(df["cgpa"]>heighest_val) | (df["cgpa"]<lowest_val)])

new_df = df[(df["cgpa"]<heighest_val) & (df["cgpa"]>lowest_val)]

#Calculating the Zscore

df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean())/df['cgpa'].std()
print(df[(df['cgpa_zscore'] > 3) | (df['cgpa_zscore'] < -3)])

new_df = df[(df['cgpa_zscore'] < 3) & (df['cgpa_zscore'] > -3)]

# # ===capping===

heighest_val = df["cgpa"].mean()+3*df["cgpa"].std()
lowest_val = df["cgpa"].mean()-3*df["cgpa"].std()

df["cgpa"] = np.where(df["cgpa"]>heighest_val,heighest_val,np.where(df["cgpa"]<lowest_val,lowest_val,df["cgpa"]))
print(df["cgpa"].describe())