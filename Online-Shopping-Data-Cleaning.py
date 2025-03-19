import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pymysql

# Load dataset
df = pd.read_csv("C:/Users/DELL/Downloads/2974735-W2_D1_Files (1)/online.csv")

# Display basic information
print(df.head())
print(df.tail())
print(df.sample(10))
print(df.info())

# Check for missing values
print("\nMissing Values Before Cleaning:\n", df.isnull().sum())

# Drop unnecessary columns
df.drop(columns=['9', '#@%', 'Unnamed: 0', 'Unnamed: 13'], inplace=True, errors='ignore')
print("\nMissing Values After Dropping Columns:\n", df.isnull().sum())

# Handle outliers
plt.figure(figsize=(10, 5))
sns.boxplot(df[['Age', 'Family size']])
plt.title("Boxplot Before Outlier Treatment")
plt.show()

# Compute IQR for Age
Q1, Q3 = df['Age'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_limit, upper_limit = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

df['Age'] = df['Age'].clip(lower_limit, upper_limit)

plt.figure(figsize=(10, 5))
sns.boxplot(df[['Age', 'Family size']])
plt.title("Boxplot After Outlier Treatment")
plt.show()

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Family size'].fillna(df['Family size'].mean(), inplace=True)
print("\nMissing Values After Imputation:\n", df.isnull().sum())

# Convert categorical data to numerical
nominal_cols = ['order_status', 'Marital Status', 'Gender', 'employment_status']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

ordinal_maps = {
    'Monthly Income': {'No Income': 1, 'Below Rs.10000': 2, '10001 to 25000': 3, '25001 to 50000': 4, 'More than 50000': 5},
    'Reviews': {'Positive': 1, 'Negative': 0},
    'Educational Qualifications': {'Uneducated': 1, 'School': 2, 'Graduate': 3, 'Post Graduate': 4, 'Ph.D': 5}
}
for col, mapping in ordinal_maps.items():
    df[col] = df[col].str.strip().map(mapping)

print(df.info())

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=np.number)), columns=df.select_dtypes(include=np.number).columns)
print(df_scaled.head())