
import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import timm

# ---------------------------
# 1️⃣ Dataset Class
# ---------------------------
class StitchedDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# ---------------------------
# 2️⃣ Dataset Paths & Labels
# ---------------------------
dataset_dir = "/content/dataset"

classes = [
    "ambulance_L", "ambulance_M", "ambulance_R",
    "FireTruck_L", "FireTruck_M", "FireTruck_R",
    "policecar_L", "policecar_M", "policecar_R",
    "carhorns_L", "carhorns_M", "carhorns_R",
    "noise"
]

img_paths, labels = [], []

for cls in classes:
    cls_folder = os.path.join(dataset_dir, cls)
    for fname in os.listdir(cls_folder):
        if fname.endswith(".png"):
            img_paths.append(os.path.join(cls_folder, fname))
            labels.append(cls)

# Encode labels
lb = LabelBinarizer()
labels_onehot = lb.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    img_paths, labels_onehot,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ---------------------------
# 3️⃣ Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = StitchedDataset(X_train, y_train, transform)
test_dataset = StitchedDataset(X_test, y_test, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------
# 4️⃣ Model Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(classes)

def get_model(model_name, num_classes):
    if model_name == "efficientnet":
        model = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=num_classes)

    elif model_name == "mobilenet":
        model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=num_classes)

    elif model_name == "ghostnet":
        model = timm.create_model("ghostnet_100", pretrained=False, num_classes=num_classes)

    else:
        raise ValueError("Unknown model")

    return model.to(device)


models_dict = {
    "efficientnet": get_model("efficientnet", num_classes),
    "mobilenet": get_model("mobilenet", num_classes),
    "ghostnet": get_model("ghostnet", num_classes)
}

# ---------------------------
# 5️⃣ Training Function
# ---------------------------
def train_model(model, train_loader, num_epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = torch.argmax(labels, dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    return model


# ---------------------------
# 6️⃣ Train All Models
# ---------------------------
for name in models_dict:
    print(f"Training {name}...")
    models_dict[name] = train_model(models_dict[name], train_loader, num_epochs=5)


# ---------------------------
# 7️⃣ Ensemble Evaluation
# ---------------------------
def ensemble_predict(models_dict, dataloader):
    all_preds = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)

        preds = []
        for model in models_dict.values():
            model.eval()
            with torch.no_grad():
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu().numpy())

        # Average ensemble
        avg_preds = np.mean(preds, axis=0)
        final_preds = np.argmax(avg_preds, axis=1)

        all_preds.extend(final_preds)
        all_labels.extend(np.argmax(labels.numpy(), axis=1))

    return np.array(all_labels), np.array(all_preds)


y_true, y_pred = ensemble_predict(models_dict, test_loader)

print("Ensemble Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))


# ---------------------------
# 8️⃣ Save Model
# ---------------------------
torch.save(models_dict["efficientnet"].state_dict(), "acoustix_edge_model.pth")
print("Model saved as acoustix_edge_model.pth")
