import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class OODDetectorDataset(Dataset):
    def __init__(self, in_domain_dir, ood_dirs, transform=None, max_ood_samples=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 1. Load In-Domain (Skin Lesions) -> Label 1
        if os.path.exists(in_domain_dir):
            in_domain_images = [
                os.path.join(in_domain_dir, f) for f in os.listdir(in_domain_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.image_paths.extend(in_domain_images)
            self.labels.extend([1] * len(in_domain_images))
        else:
            print(f"Warning: In-domain directory not found: {in_domain_dir}")
        
        # 2. Load Out-Of-Domain (COCO) -> Label 0
        ood_images = []
        for ood_dir in ood_dirs:
            if os.path.exists(ood_dir):
                ood_images.extend([
                    os.path.join(ood_dir, f) for f in os.listdir(ood_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
            else:
                print(f"Warning: OOD directory not found: {ood_dir}")
            
        random.shuffle(ood_images)
        if max_ood_samples is not None:
            ood_images = ood_images[:max_ood_samples]
            
        self.image_paths.extend(ood_images)
        self.labels.extend([0] * len(ood_images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_ood_model():
    # Use a pre-trained ResNet18 and modify the final layer for binary classification
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 0: OOD, 1: Lesion
    return model

def get_classifier_model(num_classes=7):
    # Placeholder for the 7-class lesion classifier
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_ood_detector(model, train_loader, epochs=1):
    if len(train_loader) == 0:
        print("Error: DataLoader is empty. Please check your image paths.")
        return model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for this training.")
    device = torch.device("cuda")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 50 batches so we know it's not frozen
            if (i + 1) % 50 == 0:
                print(f"  -> Epoch {epoch+1} - Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")
    
    return model

def predict_pipeline(image_path, ood_model, classifier_model, transform, class_names, classifier_threshold=0.6):
    """
    Handles the thresholding logic for the pipeline.
    """
    device = next(ood_model.parameters()).device
    ood_model.eval()
    if classifier_model:
        classifier_model.eval()
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 1. OOD Detection
    with torch.no_grad():
        ood_outputs = ood_model(input_tensor)
        ood_probabilities = torch.softmax(ood_outputs, dim=1)[0]
        
    lesion_prob = ood_probabilities[1].item()
    print(f"[{os.path.basename(image_path)}] Lesion Confidence: {lesion_prob:.4f}")
    
    # Thresholding logic from the table
    if lesion_prob <= 0.15:
        return "Out of Domain"
    elif lesion_prob < 0.85:
        return "Uncertain"
        
    # 2. 7-Class Classifier (if lesion_prob >= 0.85)
    if not classifier_model:
        return "Lesion Detected -> (No classifier provided)"
        
    with torch.no_grad():
        class_outputs = classifier_model(input_tensor)
        class_probabilities = torch.softmax(class_outputs, dim=1)[0]
        
    max_prob, predicted_idx = torch.max(class_probabilities, 0)
    max_prob = max_prob.item()
    predicted_class = class_names[predicted_idx.item()]
    
    # 3. Classifier Confidence Threshold
    if max_prob < classifier_threshold:
        return "Uncertain"
        
    return predicted_class

if __name__ == "__main__":
    # --- 1. SETTINGS & PATHS ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    skin_lesion_train_dir = r"C:\Users\abhi0\OneDrive\Desktop\Data\ISIC_2019_Training_Input\ISIC_2019_Training_Input" # Update with your path
    coco_train_dir = r"C:\Users\abhi0\OneDrive\Desktop\Data\coco2017\train2017"
    coco_val_dir = r"C:\Users\abhi0\OneDrive\Desktop\Data\coco2017\val2017"

    # --- 2. PREPARE DATASET ---
    train_dataset = OODDetectorDataset(
        in_domain_dir=skin_lesion_train_dir,
        ood_dirs=[coco_train_dir, coco_val_dir], 
        transform=transform,
        max_ood_samples=25000
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # --- 3. TRAIN/LOAD MODELS ---
    ood_model = get_ood_model()
    classifier_model = get_classifier_model(num_classes=7)
    
    print("Starting training of OOD detector...")
    # Train the model
    trained_ood_model = train_ood_detector(ood_model, train_loader, epochs=1)
    
    # Save the trained OOD model weights to the backend model directory
    backend_model_dir = os.path.join(os.path.dirname(__file__), "backend", "model")
    os.makedirs(backend_model_dir, exist_ok=True)
    ood_model_path = os.path.join(backend_model_dir, "ood_model.pth")
    torch.save(trained_ood_model.state_dict(), ood_model_path)
    print(f"OOD model saved to {ood_model_path}")
    
    # (In a real scenario, you would also train or load weights for classifier_model here)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required.")
    trained_classifier_model = classifier_model.to(torch.device("cuda"))
    
    # --- 4. TEST THE PIPELINE THRESHOLD LOGIC ---
    print("\n--- Testing Pipeline ---")
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    if os.path.exists(coco_val_dir) and len(os.listdir(coco_val_dir)) > 0:
        test_image = os.path.join(coco_val_dir, os.listdir(coco_val_dir)[0])
        result = predict_pipeline(
            test_image, 
            trained_ood_model, 
            trained_classifier_model, 
            transform, 
            class_names, 
            classifier_threshold=0.6
        )
        print(f"Final Pipeline Output for {os.path.basename(test_image)}: {result}")
    else:
        print("Could not find test image to evaluate the pipeline.")