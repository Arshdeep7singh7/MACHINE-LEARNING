import pandas as pd
data = {
    'Tid': range(1, 11),
    'Refund': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No'],
    'Marital Status': ['Single', 'Married', 'Single', 'Married', 'Divorced', 'Married', 'Divorced', 'Single', 'Married', 'Single'],
    'Taxable Income': ['125K', '100K', '70K', '120K', '95K', '60K', '220K', '85K', '75K', '90K'],
    'Cheat': ['No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print(df)
rows_selected = df.loc[[0, 4, 7, 8]]
print(rows_selected)
rows_3_to_7 = df.loc[3:7]
print(rows_3_to_7)
subset_4_8_cols_2_4 = df.iloc[4:9, 2:5]  # 4:9 because upper bound exclusive in iloc, and 2:5 for columns 2,3,4
print(subset_4_8_cols_2_4)
cols_1_to_3 = df.iloc[:, 1:4]
print(cols_1_to_3)
csv_path = 'iris.csv'  # Update this to your local path of downloaded file

iris_df = pd.read_csv(csv_path)
print(iris_df.head())
# Delete row with index 4
iris_df_dropped_row = iris_df.drop(index=4)

# Delete column with index 3
col_to_drop = iris_df.columns[3]
iris_df_dropped = iris_df_dropped_row.drop(columns=col_to_drop)

print(iris_df_dropped.head())
import pandas as pd

data = {
    'Employee_ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward'],
    'Department': ['HR', 'IT', 'IT', 'Marketing', 'Sales'],
    'Age': [29, 34, 41, 28, 38],
    'Salary': [50000, 70000, 65000, 55000, 60000],
    'Years_of_Experience': [4, 8, 10, 3, 12],
    'Joining_Date': ['2020-03-15', '2017-07-19', '2013-06-01', '2021-02-10', '2010-11-25'],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Male'],
    'Bonus': [5000, 7000, 6000, 4500, 5000],
    'Rating': [4.5, 4.0, 3.8, 4.7, 3.5]
}

df = pd.DataFrame(data)
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])  # convert Joining_Date to datetime

print(df)
print("Shape:", df.shape)
print(df.info())
print(df.describe())
print("First 5 rows:\n", df.head())
print("Last 3 rows:\n", df.tail(3))
average_salary = df['Salary'].mean()
total_bonus = df['Bonus'].sum()
youngest_age = df['Age'].min()
highest_rating = df['Rating'].max()

print(f"Average Salary: {average_salary}")
print(f"Total Bonus: {total_bonus}")
print(f"Youngest Age: {youngest_age}")
print(f"Highest Rating: {highest_rating}")
df_sorted_salary = df.sort_values(by='Salary', ascending=False)
print(df_sorted_salary)
def performance_category(rating):
    if rating >= 4.5:
        return 'Excellent'
    elif rating >= 4.0:
        return 'Good'
    else:
        return 'Average'

df['Performance_Category'] = df['Rating'].apply(performance_category)
print(df[['Name', 'Rating', 'Performance_Category']])
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)
df.rename(columns={'Employee_ID': 'ID'}, inplace=True)
print(df.columns)
experienced_employees = df[df['Years_of_Experience'] > 5]
print("Employees with more than 5 years experience:\n", experienced_employees)
it_employees = df[df['Department'] == 'IT']
print("Employees in IT department:\n", it_employees)
df['Tax'] = df['Salary'] * 0.10
print(df[['Name', 'Salary', 'Tax']])
df.to_csv('modified_employees.csv', index=False)
print("Saved modified DataFrame to 'modified_employees.csv'")
