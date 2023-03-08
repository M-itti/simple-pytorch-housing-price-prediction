from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

def load_data(data_dir=f"{os.getcwd()}/data/train.csv"):
    df = pd.read_csv(data_dir)
    X = df[['Id', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'BsmtFinType1_Unf', 
            'HasWoodDeck', 'HasOpenPorch', 'HasEnclosedPorch', 'Has3SsnPorch', 
            'HasScreenPorch', 'YearsSinceRemodel', 'Total_Home_Quality', 'LotFrontage', 
            'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
            'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
            'ScreenPorch', 'PoolArea', 'MiscVal']]

    y = df['Saleprice']

    # split data into input and target
    X = X.iloc[:, :-1].values
    y = y.values.reshape(-1, 1)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert all([x.is_cuda for x in [X_train, X_test, y_train, y_test]]), "move tensors to GPU"

    return X_train, y_train, X_test, y_test


config = {
    "lr": 0.1000,
    "epochs": 11,
    "hidden_size": 2,
    "batch_size": 38
}

# Load the data
X, y = load_data()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(config["hidden_size"], activation="relu", input_shape=(41,)),
    keras.layers.Dense(1)
])

# Compile the model
optimizer = keras.optimizers.Adam(lr=config["lr"])
model.compile(optimizer=optimizer, loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    validation_data=(X_val, y_val),
    shuffle=True
)

# Print the final loss
print("{:,.4f}".format(history.history["loss"][-1]))

