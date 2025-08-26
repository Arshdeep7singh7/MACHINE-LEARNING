import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv("AWCustomers.csv")  
print(df1.head())
print(df1.info())
df2=pd.read_csv("AWSales.csv")
print(df2.head())
print(df2.info())
combined_df = pd.merge(df1, df2, on="CustomerID", how="inner")
print(combined_df.head())
print(f"Combined shape: {combined_df.shape}")
selected_columns = [ 'YearlyIncome', 'MaritalStatus', 'Gender', 
                    'Occupation', 'NumberCarsOwned', 'BikeBuyer']
df = combined_df[selected_columns]
print(df)
data_types = {
    'YearlyIncome': 'Continuous',
    'MaritalStatus': 'Nominal',
    'Gender': 'Nominal',
    'Occupation': 'Nominal',
    'NumberCarsOwned': 'Discrete',
    'BikeBuyer': 'Binary Target'
}
print("Data Types:", data_types)
imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

numeric_cols = ['Age', 'YearlyIncome', 'NumberCarsOwned']
categorical_cols = ['MaritalStatus', 'Gender', 'Occupation']

df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])


df['AgeGroup'] = pd.cut(df['YearlyIncome'], bins=[0,50000,100000,150000,200000],
                        labels=['under 50 thousand','under lakh','under 1.5 lakhs','under 2 lakhs'])

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df[numeric_cols]),
                      columns=[col+"_std" for col in numeric_cols])

encoder = OneHotEncoder(drop='first', sparse=False)
encoded = encoder.fit_transform(df[categorical_cols + ['YearlyIncome']])
encoded_cols = encoder.get_feature_names_out(categorical_cols + ['YearlyIncome'])
df_encoded = pd.DataFrame(encoded, columns=encoded_cols)
df_processed = pd.concat([df[numeric_cols], df_std, df_encoded, df[['BikeBuyer']]], axis=1)
print(df_processed.head())
plt.figure(figsize=(12, 8))
sns.heatmap(df_processed.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

cos_sim = cosine_similarity(df_processed[numeric_cols])
print("Cosine Similarity Matrix (Numeric Features):\n", cos_sim[:5, :5])  

def simple_matching_coefficient(x, y):
    return np.mean(x == y)

customer1 = df_processed.iloc[0]
customer2 = df_processed.iloc[1]
smc = simple_matching_coefficient(customer1, customer2)
print(f"Simple Matching Coefficient between Customer 1 and 2: {smc}")

binary_data = df_encoded.values
intersection = np.logical_and(binary_data[0], binary_data[1]).sum()
union = np.logical_or(binary_data[0], binary_data[1]).sum()
jaccard = intersection / union
print(f"Jaccard Similarity between Customer 1 and 2: {jaccard}")