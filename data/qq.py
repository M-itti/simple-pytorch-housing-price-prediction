import pandas as pd

# read the train CSV file
train_df = pd.read_csv('raw_train.csv')

# create a new DataFrame with the selected columns
selected_cols = ['Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 
                 'hasbsmt', 'hasfireplace', 'LotFrontage_log', 'LotArea_log', 'MasVnrArea_log', 'BsmtFinSF1_log', 
                 'BsmtUnfSF_log', 'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'LowQualFinSF_log', 
                 'GrLivArea_log', 'BsmtFullBath_log', 'FullBath_log', 'HalfBath_log', 'BedroomAbvGr_log', 
                 'KitchenAbvGr_log', 'TotRmsAbvGrd_log', 'Fireplaces_log', 'GarageCars_log', 'GarageArea_log', 
                 'WoodDeckSF_log', 'OpenPorchSF_log', 'EnclosedPorch_log', 'ScreenPorch_log', 'PoolArea_log', 
                 'MiscVal_log', 'YearRemodAdd_log', 'TotalSF_log']
train_selected_df = train_df[selected_cols]

print(train_selected_df.dtypes)

# concatenate the original and new DataFrames
train_concat_df = pd.concat([train_df, train_selected_df], axis=1)

# write the new DataFrame to a new CSV file
train_concat_df.to_csv('train.csv', index=False)

