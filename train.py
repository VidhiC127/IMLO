import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
Batch_Size = 64
EPOCHS = 30

# Data loading and normalization
temp_set = datasets.CIFAR10("Data", train=True, download=True, transform=None)
mean = temp_set.data.mean(axis=(0,1,2)) / 255
std = temp_set.data.std(axis=(0,1,2)) / 255
print(f'Mean = {mean}, Std = {std}')

# Data augmentation for training data
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_train_set = datasets.CIFAR10("Data", train=True, download=True, transform=train_transform)
test_set = datasets.CIFAR10("Data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=Batch_Size, shuffle=False)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(full_train_set))
val_size = len(full_train_set) - train_size

train_set, val_set = random_split(full_train_set, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=Batch_Size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=Batch_Size, shuffle=False)

print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")
print(f"Test samples: {len(test_set)}")
print("Classes:", full_train_set.classes)

class CIFAR10_nn(nn.Module):
    def __init__(self):
        super(CIFAR10_nn, self).__init__()

        # 3 input channels, 32 output feature maps, 3x3 filter size
        # Padding keeps image dimensions the same
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layers reduce image size
        # Takes max value in each 2x2 region
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers connect every input to every output
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # Dropout randomly turns off 30% of neurons during training
        # Prevents nn from memorising the training data
        self.dropout = nn.Dropout(0.3)

    # Defines how data flows through the network
    def forward(self, x):

        # 1st conv layer -> reLU activation -> pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 2nd conv layer -> reLU activation -> pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 3rd conv layer -> reLU activation -> pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten 3D feature maps into a 1D vector
        x = x.view(-1, 128 * 4 * 4)

        # 1st fully connected layer -> ReLU -> dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 2nd fully connected layer -> reLU -> dropout
        x = self.dropout(F.relu(self.fc2(x)))
        # Final outout layer
        x = self.fc3(x)

        return x

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # Initialize model
    net = CIFAR10_nn()
    print(net)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(net.parameters(), lr=0.001)

    # Device configuration
    device = torch.device("cpu")
    print(f'Device: {device}')
    net.to(device)

    # Training tracking
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0
    no_improvments_num = 0

    for epoch in range(EPOCHS):
        if epoch > 0:
            torch.seed()
            np.random.seed(int(torch.randint(0, 10000, (1,))))

        # Training
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        # Calculate metrics
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluate on full training set and validation set
        train_acc = evaluate(net, train_loader, device)
        val_acc = evaluate(net, val_loader, device)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            no_improvments_num = 0
            torch.save(net.state_dict(), 'best_model.pth')
        else:
            no_improvments_num += 1
            if no_improvments_num >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                print(f'  Loss: {epoch_loss:.4f}  Training Accuracy: {train_acc:.2f}%  Validation Accuracy: {val_acc:.2f}%')
                break

        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Loss: {epoch_loss:.4f}  Training Accuracy: {train_acc:.2f}%  Validation Accuracy: {val_acc:.2f}%')
        print("")

    '''# Final evaluation on test set
    test_accuracy = evaluate(net, test_loader, device)
    print(f'Final Test Accuracy: {test_accuracy:.2f}%')

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'g-', label='Training Accuracy')
    plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='k', linestyle='--', label=f'Final Test: {test_accuracy:.2f}%')
    plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()'''

if __name__ == "__main__":
    print("Training:")
    main()
