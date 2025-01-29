from dataloader.data_loader import PlantDiseaseDataset, load_images
from model.model import PlantDiseaseModel
from utils.plot import *
from executors.train import train_model, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import yaml
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the YAML file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

print(config)

if config["training"]["use_wandb"]:
    exp_name = config["model"]["dataset"] + "_" + config["model"]["name"]
    wandb.init(project=exp_name)
    wandb.config = config

# Load data
image_paths, labels = load_images(config["data"]["directory_root"])

# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train, validation, and test splits
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
)
valid_paths, test_paths, valid_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),           # Data augmentation for training
    transforms.RandomRotation(30),              # Random rotation for variability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

valid_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),              # Consistent resizing for validation/test
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as training
])

# Create datasets with appropriate transformations
train_dataset = PlantDiseaseDataset(train_paths, train_labels, transform=train_transform)
valid_dataset = PlantDiseaseDataset(valid_paths, valid_labels, transform=valid_test_transform)
test_dataset = PlantDiseaseDataset(test_paths, test_labels, transform=valid_test_transform)

# Create dataloaders
batch_size = config["training"]["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for training
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) # No shuffle for validation/test
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # No shuffle for test

lr = config["model"]["hyperparameters"]["optimizer"]["learning_rate"]
num_classes = len(label_encoder.classes_)
model = PlantDiseaseModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

n_epochs = config["training"]["epochs"]

early_stopping = EarlyStopping(patience=4, min_delta=0.01, save_path=config["training"]["early_stopping"]["save_path"])
train_losses, valid_losses, valid_accuracies = train_model(
    model, train_loader, valid_loader, criterion, optimizer, epochs=n_epochs, early_stopping=early_stopping, wandb_log=config["training"]["use_wandb"]
)

plot_learning_curve(train_losses, valid_losses, valid_accuracies)

