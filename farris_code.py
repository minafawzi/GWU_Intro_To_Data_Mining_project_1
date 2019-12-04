
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import sys
import seaborn as sns
import pandas as pd

filePath = 'C:\\Users\\farri\\PycharmProjects\\untitled\\venv\\linkedin_data.csv'
df = pd.read_csv(filePath, encoding='latin1')
df.shape
df.head()
df = df[['age', 'ethnicity', 'gender', 'mouth_close', 'smile', 'n_followers']]
df.corr(method='kendall')

plt.figure(figsize=(7, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
df_pairplot = df.dropna()
sns.pairplot(df_pairplot, height=1.5)


