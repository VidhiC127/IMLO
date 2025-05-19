import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from train import CIFAR10_nn, evaluate

def main():
    # Device configuration
    device = torch.device("cpu")
    
    # Load model
    model = CIFAR10_nn().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Load test data
    temp_set = datasets.CIFAR10("Data", train=True, download=True, transform=None)
    mean = temp_set.data.mean(axis=(0,1,2)) / 255
    std = temp_set.data.std(axis=(0,1,2)) / 255
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_set = datasets.CIFAR10("Data", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    # Evaluate
    test_accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

if __name__ == "__main__":
    print("Testing:")
    main()