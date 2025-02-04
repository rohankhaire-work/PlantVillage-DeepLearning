from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataloader.data_loader import PlantDiseaseDataset, load_images
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader.data_loader import load_images
from model.model import PlantDiseaseModel
import torch.optim as optim
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import tqdm
import yaml
 
def evaluate_model(model, data_loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(data_loader)
    accuracy = correct / total * 100
    return val_loss, accuracy

def train_model(config):
    train_losses, valid_losses, valid_accuracies = [], [], []
    epochs = 10

    model = PlantDiseaseModel(config["classes"]).to(device)

    # Create dataloaders
    train_loader = DataLoader(config["train_dataset"], batch_size=int(config["batch_size"]), shuffle=True)  # Shuffle for training
    valid_loader = DataLoader(config["val_dataset"], batch_size=int(config["batch_size"]), shuffle=True) # shuffle for validation/test

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{epochs}", total=len(train_loader))

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})

        # Record training loss
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total * 100
        train_losses.append(train_loss)

        # Validation step
        val_loss, val_accuracy = evaluate_model(model, valid_loader, criterion)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

    return train_losses, valid_losses, valid_accuracies


if __name__ == "__main__":
    # Read the YAML file
    with open('config/config.yaml', 'r') as file:
        params = yaml.safe_load(file)
    
    # Load data
    image_paths, labels = load_images(params["data"]["directory_root"])

    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    # Train, validation, and test splits
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),           # Data augmentation for training
    transforms.RandomRotation(30),              # Random rotation for variability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),              # Consistent resizing for validation/test
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as training
    ])

    # Create datasets with appropriate transformations
    train_dataset = PlantDiseaseDataset(train_paths, train_labels, transform=train_transform)
    valid_dataset = PlantDiseaseDataset(valid_paths, valid_labels, transform=valid_test_transform)
    test_dataset = PlantDiseaseDataset(test_paths, test_labels, transform=valid_test_transform)

    # hyperparameters to tune
    config = {
        "momentum": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
        "lr": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
        "train_dataset": train_dataset,
        "val_dataset": valid_dataset,
        "classes": num_classes,
    }

    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=10,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
