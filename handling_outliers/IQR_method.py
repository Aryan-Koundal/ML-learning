import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("csv_files/placement.csv")
print(df.describe())

percentile_25 = df["placement_exam_marks"].quantile(0.25)
percentile_75 = df["placement_exam_marks"].quantile(0.75)

iqr = percentile_75 - percentile_25

upper_limit = percentile_75 + 1.5 * iqr
lower_limit = percentile_25 - 1.5 * iqr

print("Upper Limit = ",upper_limit)
print("Lower Limit = ",lower_limit)

print(df[df["placement_exam_marks"]>upper_limit])

# # ===trimming===
new_df = df[df["placement_exam_marks"]<upper_limit]

print(new_df.head())

print(new_df.shape)

# Comparing

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])

plt.show()


# # ===capping===

df["placement_exam_marks"] = np.where(df["placement_exam_marks"]>upper_limit,upper_limit,np.where(df["placement_exam_marks"]<lower_limit,lower_limit,df["placement_exam_marks"]))

new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)

# Comparing

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])

plt.show()