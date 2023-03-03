import pandas as pd

df = pd.read_csv('train.csv')

# Read CSV file into a pandas dataframe

# Print the first 5 rows of the dataframe
print(df.head())

# Print the last 5 rows of the dataframe
print(df.tail())

# Print the shape of the dataframe (number of rows, number of columns)
print(df.shape)

# Print the column names
print(df.columns)

print(df.dtypes.to_string())

# Print the data types of each column

# Print summary statistics of the numerical columns
print(df.describe())

# Print the number of missing values in each column
print(df.isnull().sum())
