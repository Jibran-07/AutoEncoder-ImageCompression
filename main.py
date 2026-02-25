import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size=784, hidden_size=32):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded


transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = SimpleAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.view(-1, 784))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")


model.eval()
test_loss = 0.0
with torch.no_grad():
    for data, _ in test_loader:
        outputs = model(data)
        loss = criterion(outputs, data.view(-1, 784))
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

sample_data = next(iter(test_loader))[0][0]
with torch.no_grad():
    reconstructed = model(sample_data.unsqueeze(0))

original_img = sample_data.view(28, 28)
reconstructed_img = reconstructed.view(28, 28)


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.show()