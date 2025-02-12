import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import csv
from tqdm import tqdm  # Import tqdm for progress bars


# =========================================
# STEP 1: Define On-the-Fly Augmentation Dataset
# =========================================

class AugmentedDataset(Dataset):
    def __init__(self, root_dir, base_transform=None, augmentations=None, num_augments=10):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.augmentations = augmentations
        self.base_transform = base_transform
        self.num_augments = num_augments

    def __len__(self):
        return len(self.dataset) * self.num_augments

    def __getitem__(self, idx):
        original_idx = idx // self.num_augments  # Ensure each image is sampled exactly 10 times
        img, label = self.dataset[original_idx]

        if self.augmentations:
            img = self.augmentations(img)  # Apply random augmentations

        if self.base_transform:
            img = self.base_transform(img)  # Convert to tensor and normalize

        return img, label


# Define Augmentation Pipeline (apply before conversion to tensor)
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),
])

# Data Transforms for Training and Validation
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets (Using AugmentedDataset for Training)
train_dataset = AugmentedDataset("BDMA7_project_files/train_images", base_transform=base_transform,
                                 augmentations=augmentation)
val_dataset = datasets.ImageFolder('BDMA7_project_files/val_images', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =========================================
# STEP 2: Load Pretrained Model (DenseNet201)
# =========================================

# Load pre-trained DenseNet201 model
model = models.densenet201(weights="IMAGENET1K_V1")

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the final classifier layer for our number of classes
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(train_dataset.dataset.classes))  # Adjust for number of classes

# Send model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    # Wrapping train_loader with tqdm to show progress
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    scheduler.step()

    # Training accuracy
    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

# Evaluation on validation set with tqdm
model.eval()
correct = 0
total = 0
with torch.no_grad():
    # Wrapping val_loader with tqdm to show progress
    for inputs, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Save the model after training
torch.save(model.state_dict(), 'densenet201_model.pth')
print("Model saved as 'densenet201_model.pth'")


'''
-------------------------------------------------------------------------
'''

# Custom dataset for loading unlabeled images
class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # List all image files in the directory
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image file path
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Open image and ensure it's in RGB format

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]  # Return the image and its filename


# Test transforms (for the unlabeled images)
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the custom dataset for the test set
test_dataset = UnlabeledImageDataset('BDMA7_project_files/test_images/mistery_cat', transform=test_transforms)

# DataLoader for the test set
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the model
model = models.densenet201(weights="IMAGENET1K_V1")  # Load the pretrained model
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(train_dataset.classes))  # Adjust for number of classes
model.load_state_dict(torch.load('densenet201_model.pth'))  # Load the trained model
model = model.to(device)
model.eval()

# Predict on unlabeled images
predictions = []
with torch.no_grad():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Collect predictions with corresponding filenames
        for filename, pred in zip(filenames, predicted):
            predictions.append((filename, pred.item()))

# Print predictions for all test images
#for filename, pred in predictions:
#    print(f"Image: {filename}, Predicted Class: {train_dataset.classes[pred]}")


'''
TEST PART
---------------------------------------------------------------------------------
'''

# Read the test predictions from the CSV file
test_predictions_df = pd.read_csv('test_predictions.csv')

# Create a dictionary for quick access to true class labels from the CSV file
true_labels_dict = dict(zip(test_predictions_df['Image'], test_predictions_df['Predicted Class']))

# List to store comparison results
correct_predictions = 0
total_predictions = 0

# Compare the model's predictions with the true labels
for filename, predicted_class in predictions:
    true_class = true_labels_dict.get(filename)  # Get the true class from the CSV

    if true_class:  # If true class is found in the CSV file
        total_predictions += 1
        if true_class == train_dataset.classes[predicted_class]:  # Compare the predicted class with the true class
            correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")

# If you'd like to see detailed comparison
for filename, predicted_class in predictions:
    true_class = true_labels_dict.get(filename)
    if true_class:
        print(f"Image: {filename}, Predicted Class: {train_dataset.classes[predicted_class]}, True Class: {true_class}")

'''
---------------------------------------------------------
'''

# Assuming predictions are stored in a list of tuples (filename, predicted_class_index)
# Example: predictions = [('000c02a0-3687-4c0c-bdfc-394b56134ac6.jpg', 0), ...]

# Read the test predictions from the CSV file
test_predictions_df = pd.read_csv('test_predictions.csv')

# Create a dictionary for quick access to true class labels from the CSV file
true_labels_dict = dict(zip(test_predictions_df['Image'], test_predictions_df['Predicted Class']))

# List to store comparison results and submission
correct_predictions = 0
total_predictions = 0
submission_data = []

# Compare the model's predictions with the true labels
for filename, predicted_class in predictions:  # Assuming predictions is your list of (filename, predicted_class)
    true_class = true_labels_dict.get(filename)  # Get the true class from the CSV

    if true_class:  # If true class is found in the CSV file
        total_predictions += 1
        predicted_class_label = train_dataset.classes[predicted_class]

        # Print if the comparison is true or false
        if true_class == predicted_class_label:
            correct_predictions += 1
            print(f"Image: {filename}, Prediction: {predicted_class_label}, True Class: {true_class} - Correct")
        else:
            print(f"Image: {filename}, Prediction: {predicted_class_label}, True Class: {true_class} - Incorrect")

    # Add to submission data for Kaggle
    submission_data.append([filename, predicted_class_label])

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")


'''


'''

# Create a DataFrame for Kaggle submission
submission_df = pd.DataFrame(submission_data, columns=['path','class_idx'])

# Save the DataFrame as a CSV for Kaggle submission
submission_df.to_csv('kaggle_submission_densenet201_ready_to_be_finalized.csv', index=False)

# Dictionary mapping bird names to numbers
bird_dict = {
    'Groove_billed_Ani': 0,
    'Red_winged_Blackbird': 1,
    'Rusty_Blackbird': 2,
    'Gray_Catbird': 3,
    'Brandt_Cormorant': 4,
    'Eastern_Towhee': 5,
    'Indigo_Bunting': 6,
    'Brewer_Blackbird': 7,
    'Painted_Bunting': 8,
    'Bobolink': 9,
    'Lazuli_Bunting': 10,
    'Yellow_headed_Blackbird': 11,
    'American_Crow': 12,
    'Fish_Crow': 13,
    'Brown_Creeper': 14,
    'Yellow_billed_Cuckoo': 15,
    'Yellow_breasted_Chat': 16,
    'Black_billed_Cuckoo': 17,
    'Gray_crowned_Rosy_Finch': 18,
    'Bronzed_Cowbird': 19
}

# Function to replace bird names with numbers
def replace_bird_names(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Assuming bird names are in the second column (index 1)
            if row[1] in bird_dict:
                row[1] = bird_dict[row[1]]  # Replace bird name with number
            writer.writerow(row)


# Process the file
replace_bird_names('kaggle_submission_densenet201_ready_to_be_finalized.csv', 'kaggle_submission_densenet201_FINAL.csv')

print("Kaggle submission file 'kaggle_submission_densenet201_FINAL.csv' has been saved and is ready to be submitted")
