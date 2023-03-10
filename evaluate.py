import torch
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available else "cpu"

def load_data(data_dir=f"{os.getcwd()}/data/train.csv"):
    df = pd.read_csv(data_dir)


    X = df[['Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 
            'hasbsmt', 'hasfireplace', 'LotFrontage_log', 'LotArea_log', 'MasVnrArea_log', 'BsmtFinSF1_log', 
            'BsmtUnfSF_log', 'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'LowQualFinSF_log', 
            'GrLivArea_log', 'BsmtFullBath_log', 'FullBath_log', 'HalfBath_log', 'BedroomAbvGr_log', 
            'KitchenAbvGr_log', 'TotRmsAbvGrd_log', 'Fireplaces_log', 'GarageCars_log', 'GarageArea_log', 
            'WoodDeckSF_log', 'OpenPorchSF_log', 'EnclosedPorch_log', 'ScreenPorch_log', 'PoolArea_log', 
            'MiscVal_log', 'YearRemodAdd_log', 'TotalSF_log',
            'Id', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'BsmtFinType1_Unf', 
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

X_train, y_train, X_val, y_val = load_data()


config = {
        "lr": 0.1000,
        "epoches": 100,
        "hidden_size": 20,
        "batch_size": 36
}


X_train, y_train, X_val, y_val = load_data()

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


class Net(nn.Module):
    def __init__(self, hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(76, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate neural network
net = Net(config["hidden_size"]).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
rmse_loss = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

# Train the neural network
for epoch in range(config["epoches"]):
    # Training loop
    train_loss = 0
    train_total_mape = 0
    for i, batch in enumerate(train_dataloader): # 36
        X_batch, y_batch = batch
        # Forward pass
        outputs = net(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Calculate MAPE
        with torch.no_grad():
            mape = torch.mean(torch.abs((outputs - y_batch) / y_batch)) * 100
        train_total_mape += mape.item()

        # Print progress
        print("Iteration {}: running loss {:.4f}".format(i, loss.item()))

   # validation loop
    val_loss = 0
    val_total_mape = 0
    net.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            X_batch, y_batch = batch
            # Forward pass
            outputs = net(X_batch)

            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # Calculate MAPE
            mape = torch.mean(torch.abs((outputs - y_batch) / y_batch)) * 100
            val_total_mape += mape.item()

    # Compute average loss over training and validation sets
    val_avg_mape = val_total_mape / len(val_dataloader)
    train_avg_mape = train_total_mape / len(train_dataloader)
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    rmse = torch.sqrt(rmse_loss(outputs, y_batch))

    print("rmse", rmse.item())
    print("train_loss", train_loss)
    print("val_loss", val_loss)

    print("Epoch {}: train loss {:.4f}, val loss {:.4f}, val MAPE {:.2f}%, train MAPE {:.2f}".format(epoch+1, train_loss, val_loss, val_avg_mape, train_avg_mape))
