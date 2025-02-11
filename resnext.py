
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


# Data transforms (augmentation)
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('BDMA7_project_files/train_images', transform=train_transforms)
val_dataset = datasets.ImageFolder('BDMA7_project_files/val_images', transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained models.resnext101_32x8d() model (largest available DenseNet model)
model = models.resnext101_32x8d(weights='ResNeXt101_32X8D_Weights.DEFAULT')  # or weights=models.resnext101_32x8d()_Weights.IMAGENET1K_V1 if available


# Freeze the parameters of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer to match the number of classes
num_ftrs = model.fc.in_features
# Modify the final fully connected layer to match the number of classes
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

# Send model to device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy}%")

# Evaluation on validation set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f"Validation Accuracy: {val_accuracy}%")

# Save the model after training
torch.save(model.state_dict(), 'models.resnext101_32x8d()_model.pth')
print("Model saved as 'models.resnext101_32x8d()_model.pth'")

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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = models.resnext101_32x8d(weights='ResNeXt101_32X8D_Weights.DEFAULT')  # Load the pretrained model

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

model.load_state_dict(torch.load('models.resnext101_32x8d()_model.pth'))  # Load the trained model
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
for filename, pred in predictions:
    print(f"Image: {filename}, Predicted Class: {train_dataset.classes[pred]}")


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
# You can modify how predictions are stored as per your actual code.

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
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")



# Create a DataFrame for Kaggle submission
submission_df = pd.DataFrame(submission_data, columns=['Image', 'Predicted Class'])

# Dictionary for mapping bird names to numbers
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

with open('kaggle_submission.csv', mode='r') as infile, open('kaggle_submission_final.csv', mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        bird_name = row[1]  # Get the second part (bird name)
        if bird_name in bird_dict:
            row[1] = bird_dict[bird_name]  # Replace bird name with number
        writer.writerow(row)

# Save the DataFrame as a CSV for Kaggle submission
submission_df.to_csv('kaggle_submission_preparation.csv', index=False)
print("Kaggle submission file 'kaggle_submission_preparation.csv' has been saved.")

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
replace_bird_names('kaggle_submission_preparation.csv', 'kaggle_ready_resnext.csv')

print("File processed successfully! Output saved as kaggle_ready_resnext.csv")
