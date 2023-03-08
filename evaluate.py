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

X_train, y_train, X_val, y_val = load_data()


config = {
        "lr": 0.1000,
        "epoches": 11,
        "hidden_size": 2,
        "batch_size": 38
}


X_train, y_train, X_val, y_val = load_data()

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


class Net(nn.Module):
    def __init__(self, hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(41, hidden)
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

        print(("loss", loss.item()))

        # Print progress
        print("Iteration {}: running loss {:.4f}".format(i, loss.item()))

   # validation loop
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            X_batch, y_batch = batch
            # Forward pass
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()
            #batch_size = X_batch.size(0)
			#num_samples += batch_size
			#rmse_batch = torch.sqrt(mse(outputs, y_batch))
			#rmse += rmse_batch.item()**2 * batch_size

    # Compute average loss over training and validation sets
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    
    rmse = torch.sqrt(rmse_loss(outputs, y_batch))

    print("rmse", rmse.item())
    print("train_loss", train_loss)
    print("val_loss", val_loss)

    print(f"Epoch {epoch}: train_loss={train_loss:,.4f}, val_loss={val_loss:,.4f}")

print("{:,.4f}".format(loss.item()))
