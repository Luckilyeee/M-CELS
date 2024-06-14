from pyts.datasets import fetch_ucr_dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, TensorDataset
from fcn_pytorch_model import FCN

# dataset = ['Coffee', 'ECG200', 'GunPoint', 'TwoLeadECG'， 'CBF']
# [1.0, 0.88, 0.99, 0.99, 0.99]
dataset = ['Coffee']
softmax_fn = torch.nn.Softmax(dim=-1)

acc = []
for data in dataset:
    print("start: ", data)
    data_train, data_test, target_train, target_test = fetch_ucr_dataset(data, use_cache=True, data_home=None, return_X_y=True)
    print(data_train.shape)
    data_train = data_train.reshape(data_train.shape[0], 1, data_train.shape[1])
    data_test = data_test.reshape(data_test.shape[0], 1, data_test.shape[1])

    encoder = LabelEncoder()
    target_train = encoder.fit_transform(target_train)
    target_test = encoder.fit_transform(target_test)

    train_data = torch.tensor(data_train, dtype=torch.float32)
    train_targets = torch.tensor(target_train, dtype=torch.long)
    test_data = torch.tensor(data_test, dtype=torch.float32)
    test_targets = torch.tensor(target_test, dtype=torch.long)

    # Create a PyTorch dataset and dataloader for training data
    train_dataset = TensorDataset(train_data, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Split the training data into training and validation sets
    val_ratio = 0.2  # Define the ratio of validation data
    val_size = int(val_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    # Instantiate the FCN model
    # print("train_data_shape: ", train_data.dataset.tensors[0].shape)
    # train_data_shape:  torch.Size([28, 286])
    input_size = train_data.dataset.tensors[0].shape[1]  # Adjust the input size based on your data
    # print("input_size", input_size)
    # input_size: 286
    num_classes = len(torch.unique(train_targets))
    model = FCN(input_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            # print("inputs.shape", inputs.shape)
            outputs = model(inputs)
            # print(outputs, outputs.shape)   (16, 2)
            # break
            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute the validation loss
                val_loss += criterion(outputs, targets).item()

                # Compute the validation accuracy
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item()}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

        # Save the best model based on validation loss
        save_dir = "models/"

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the model in the specified directory
        model_path = os.path.join(save_dir, data + "_best_model.pth")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    # Load the saved model
    saved_model = FCN(input_size, num_classes)
    saved_model.load_state_dict(torch.load(model_path))
    saved_model.to(device)

    # Evaluate the model on the test set
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    # print(test_data.shape, test_targets.shape)

    saved_model.eval()
    with torch.no_grad():
        outputs = saved_model(test_data)
        # print(outputs,outputs.shape)
        print(softmax_fn(outputs))
        a, predicted = torch.max(outputs, 1)
        # print(predicted)
        correct = (predicted == test_targets).sum().item()
        total = test_targets.size(0)
        accuracy = correct / total
        acc.append(accuracy)
        print("accuracy: ", accuracy)
print(acc)
