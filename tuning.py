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
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler



device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 40

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


def train(config):
    with mlflow.start_run():
        X_train, y_train, X_val, y_val = load_data()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

                mlflow.log_metric("loss", loss.item())
                mlflow.pytorch.log_model(net, "model")

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

            # Compute average loss over training and validation sets
            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        tune.report(loss=loss.item(), step=epoch+1)

        return  {"loss": val_loss}


if __name__ == "__main__":
    exp_id = mlflow.create_experiment("house_price_prediction")
    mlflow.set_experiment(exp_id)
    ray.init(num_gpus=1)   

    sha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            max_t=100,
            grace_period=10,
            reduction_factor=3,
            brackets=1,
)   

    reporter = CLIReporter(max_progress_rows=20, metric_columns=["loss"], print_intermediate_tables=5)
        
    tune_config = TuneConfig(
            metric="loss",
            mode="min",
            num_samples=4,
            search_alg=OptunaSearch(),
            scheduler=sha_scheduler
            )
    tuner = tune.Tuner(
        tune.with_resources(train, {"gpu": 1}),
        tune_config=tune_config,
        run_config=RunConfig(
            verbose=3
    ),
        param_space={
            "lr": 0.1000,
            "hidden_size": tune.choice([1,2,3]),
            "epoches": tune.choice([8, 7, 11]),
            "batch_size": tune.choice([32,36,38]),
            "num_layers": tune.choice([1,2,3])
        }
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(metric="loss", mode="min")
    print("Best result", best_result)

    best_loss = best_result.metrics
    print(best_loss['loss'])

    best_config = best_result.config 
    print(best_config)
