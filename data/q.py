
import pandas as pd

# Read in the CSV file
df = pd.read_csv("raw_train.csv")

# List of columns to drop
cols_to_drop = ['Id', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'BsmtFinType1_Unf', 
                'HasWoodDeck', 'HasOpenPorch', 'HasEnclosedPorch', 'Has3SsnPorch', 
                'HasScreenPorch', 'YearsSinceRemodel', 'Total_Home_Quality', 'LotFrontage', 
                'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
                'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                'ScreenPorch', 'PoolArea', 'MiscVal']


# Drop the specified columns and print the resulting dataframe
df.drop(cols_to_drop, axis=1, inplace=True)
print(df.columns)

