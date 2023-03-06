import torch
import torch.nn as nn
import pandas as pd
import ray
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
import mlflow 
import mlflow.pytorch
from torch.utils.data import DataLoader
from ray.tune import CLIReporter
from ray.air.config import RunConfig
from ray.tune import TuneConfig
import os
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(data_dir=f"{os.getcwd()}/train.csv"):
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


def train_mnist(config):
    # TODO: write a test for ray tune (test datasets are missing)
    X_train, y_train, _, _ = load_data()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    batch_size = 40
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    # Define neural network architecture
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
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    # Train the neural network
    for epoch in range(config["epoches"]):
        for i, batch in enumerate(dataloader): # 36
            X_batch, y_batch = batch
            # Forward pass
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric("loss", loss.item())
            mlflow.pytorch.log_model(net, "model")

            # Print progress
            print("Iteration {}: running loss {:.4f}".format(i, loss.item()))
        print(f"Finished epoch {epoch}")


    tune.report(loss=loss.item(), step=epoch+1)


    return  {"loss": loss.item()}


if __name__ == "__main__":
    ray.init(num_gpus=1)

    reporter = CLIReporter(max_progress_rows=20, metric_columns=["loss"], print_intermediate_tables=5)
    
    tune_config = TuneConfig(
            metric="loss",
            mode="min",
            num_samples=4
    )

    tuner = tune.Tuner(
        tune.with_resources(train_mnist, {"gpu": 1}),
        tune_config=tune_config,
        run_config=RunConfig(
            verbose=3
    ),
        param_space={
            "lr": tune.grid_search([0.001, 0.01, 0.1]),
            "hidden_size": tune.choice([1,2,3]),
            "epoches": tune.choice([5, 10, 11])
        }
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(metric="loss", mode="min")
    print("Best result", best_result)

    best_loss = best_result.metrics
    print(best_loss['loss'])

    best_config = best_result.config 
    print(best_config)
