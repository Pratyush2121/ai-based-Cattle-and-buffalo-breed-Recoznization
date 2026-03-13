# main.py (upgraded)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler

# -------------------------------
# Step 1: Paths and parameters
# -------------------------------
DATA_DIR = "data/processed"   # train/val folders
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Step 2: Data transforms & loaders
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = train_dataset.classes
print("Classes:", classes)

# -------------------------------
# Step 3: Load Pretrained Model
# -------------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

# -------------------------------
# Step 4: Weighted Loss (optional)
# -------------------------------
class_counts = [len(os.listdir(os.path.join(DATA_DIR, "train", c))) for c in classes]
class_weights = torch.tensor([1.0/count for count in class_counts]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------------------
# Step 5: Training Loop
# -------------------------------
best_acc = 0.0
train_acc_list = []
val_acc_list = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)
    train_acc_list.append(epoch_acc)

    # Validation
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            val_corrects += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_corrects / len(val_dataset)
    val_acc_list.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {epoch_loss:.4f} "
          f"Train Acc: {epoch_acc:.4f} "
          f"Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

    scheduler.step()

print("Training complete. Best Val Acc:", best_acc)

# -------------------------------
# Step 6: Save final train/val accuracy for plotting
# -------------------------------
import matplotlib.pyplot as plt

plt.plot(range(EPOCHS), train_acc_list, label="Train Acc")
plt.plot(range(EPOCHS), val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
