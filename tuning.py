import torch
import torch.nn as nn
import pandas as pd
import ray
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
import mlflow 
import mlflow.pytorch

def train_mnist(config):
    #torch.save(net.state_dict(), "model.pt")
    with mlflow.start_run():
        mlflow.log_params(config)

        # Load data
        df = pd.read_csv('/home/mahdi/project/data/train.csv')
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

        # Convert data to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        # Define neural network architecture
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(41, 1)
                self.fc2 = nn.Linear(1, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        # Instantiate neural network
        net = Net()

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

        # Train the neural network
        num_epochs = 1
        for epoch in range(num_epochs):
            # Forward pass
            outputs = net(X)
            loss = criterion(outputs, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric("loss", loss.item())
            mlflow.pytorch.log_model(net, "model")


            # Print progress
            if (epoch+1) % 100 == 0:
                tune.report(loss=loss.item(), step=epoch+1)

        return  {"loss": loss.item()}

if __name__ == "__main__":
    #mlflow.set_tracking_uri('my-experiment')
    mlflow.set_tracking_uri('http://127.0.40:5000')
    mlflow.set_experiment("my-experiment")
    #mlflow.create_experiment("my-experiment")
    
    ray.init(num_cpus=4)

    tuner = tune.Tuner(
        train_mnist,
        tune_config=tune.TuneConfig(mode="min", metric="loss"),
        param_space={
            "lr": tune.grid_search([0.001, 0.01, 0.1]),
            }
        )
    

    results = tuner.fit()
    
    best_result = results.get_best_result( 
                metric="loss", mode="min")
    print("Best result", best_result)

    best_loss = best_result.metrics
    print(best_loss['loss'])

    best_config = best_result.config 
    print(best_config)

    # Log best result
    mlflow.log_metric("loss", best_loss['loss'])
    mlflow.log_params(best_config)
