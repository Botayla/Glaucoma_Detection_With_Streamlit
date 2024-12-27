# import libraries 
import streamlit as st
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
from tqdm.notebook import tqdm
import warnings 
warnings.filterwarnings("ignore")
st.title("Glaucoma Detection With CNN")
#--------------------------------------------------------------------------------------------------------
class cnnmodel(nn.Module):                                    
    def __init__(self):                                        # initializes the model 
        super().__init__()                                     # calls the parent class nn.Module's constructor to inherit its properties and methods.        
        self.conv_network = nn.Sequential(                                                             # stacks convolutional layers  for feature extraction 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),   # conv layer with 3 input channels,32 output channels,3x3 kernel size,stride of 1,and padding of 1.>> 3*224*224->32*224*224
            nn.ReLU(),                                                                       #  Activation function to introduce non-linearity.
            nn.BatchNorm2d(32),                                                              # Normalizes the output of the convolution layer for better convergence.
            nn.MaxPool2d(kernel_size=2, stride=2),                  # downsampling dimensions by a factor of 2 .   >>  32x224x224 -> 32x112x112           
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 32x112x112 -> 64*112*112
            nn.ReLU(),                                                                    
            nn.BatchNorm2d(64),                                                           
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x112x112 -> 64x56x56            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),   # 64x112x112 -> 64x56x56
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),              
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 64x28x28  -> 128x28x28 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x28x28 -> 128x14x14            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 128x14x14 ->  256x14x14
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2) )   # 256x14x14 -> 256x7x7          
        # Fully connected layers
        self.fc_network = nn.Sequential(
            nn.Flatten(),                         #flatten to 1D vector
            nn.Linear(256 * 7 * 7, 1024),         # input 256*7*7 output 1024 
            nn.ReLU(),
            nn.Dropout(0.6),                     # dropout 60 % from the connection to prevent overfitting 
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2))  # 2 classes (positive/negative)        
    def forward(self, x):
        x = self.conv_network(x)  # Feature extraction
        x = self.fc_network(x)  # Classification
        return x
model = cnnmodel()
#--------------------------------------------------------------------------------------------------------
# Define transformations for training and testing
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                # Resize images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),       # Flip horizontally with a probability of 50%
    transforms.RandomVerticalFlip(p=0.2),         # Flip vertically with a probability of 20%
    transforms.RandomRotation(degrees=15),        # Random rotation between -15 and +15 degrees
    transforms.ColorJitter(brightness=0.2,        # Adjust brightness randomly by ±20%
                           contrast=0.2,          # Adjust contrast randomly by ±20%
                           saturation=0.2,        # Adjust saturation randomly by ±20%
                           hue=0.1),              # Adjust hue randomly by ±10%
    transforms.RandomAffine(degrees=0,            # Random affine transformation (rotation disabled)
                            shear=10,             # Shear transformations up to 10 degrees
                            scale=(0.8, 1.2)),    # Scale images between 80% to 120%
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Add perspective distortion
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),   # Random crop and resize
    transforms.ToTensor(),                        # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                # Resize to 224x224
    transforms.ToTensor(),                        # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])
#--------------------------------------------------------------------------------------------------------
st.markdown("### Show some Images :")
train_path =  "H:\data\Fundus_Train_Val_Data\Train"  # path to train dataset 
validation_path =  "H:\data\Fundus_Train_Val_Data\Validation"  # path to validation dataset
images = ['001.jpg', '002.jpg', '003.jpg', '004.jpg', '005.jpg']  
cols = st.columns(len(images)) 
# Display the images using matplotlib
for i, image_name in enumerate(images):
    image_path = os.path.join(train_path, image_name)
    img = cv2.imread(image_path)                                  # Read the image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                   # Convert BGR to RGB for matplotlib
    with cols[i]:
        st.image(img, caption=image_name, use_column_width=True)                                    # Create a subplot for each image

df =  pd.read_csv("H:\data\glaucoma.csv")  # read csv file have the images and labels 
st.markdown("The distribution of Classes :")
st.write(df["Glaucoma"].value_counts())   # Distribution of diagnosis in our dataset: 0: Negative, 1: Positive
# Plot Class Distribution
plt.figure(figsize=(8, 6))
df["Glaucoma"].value_counts().plot(kind="bar", color=["pink", "yellow"])
plt.xlabel("Class", fontsize=14)
plt.ylabel("Number of Images", fontsize=14)
plt.title("Class Distribution in Dataset", fontsize=16)
plt.xticks(rotation=0)
st.pyplot(plt)
all_images = []                              # empty list to have all images (train , validation ) in one list 
#save all images 
for filename in os.listdir(train_path):      # Looping through train folder to append images in all_images list
    all_images.append(os.path.join(train_path, filename))     # add the image to all_images list 

for filename in os.listdir(validation_path):  ## Looping through validation folder to append images in all_images list
    all_images.append(os.path.join(validation_path, filename))    # add the image to all_images list 
#--------------------------------------------------------------------------------------------------------
train_ratio = 0.7                 # make the train set be 70% of all images 
validation_ratio = 0.15           # make the train set be 15% of all images 
test_ratio = 0.15                 # make the train set be 15% of all images 
np.random.shuffle(all_images)     # Shuffle the images
# Split indices
total_images = len(all_images)                                            # get the total images in the data set 
train_split = int(train_ratio * total_images)                             # get the number of images should be in the train set  
validation_split = train_split + int(validation_ratio * total_images)     # get the number of images should be in the validation set  
train_images = all_images[:train_split]                                    # save the images fro tarin set from the beginning  
validation_images = all_images[train_split:validation_split]               # save the images frovalidation  set from the beginning  
test_images = all_images[validation_split:]                                #save the images fro tarin set from the beginning  
# Extract labels from the DataFrame
def get_labels(image_list, df):                                             # Defining the function for extract the label of the image
    labels = []  # save the labels of the images 
    for image_path in image_list:                                           # Looping through the images from all_images list to get the label
        image_name = os.path.basename(image_path)                           # get the name of the image 
        label = df[df["Filename"] == image_name]["Glaucoma"].values[0]      # get the row matching the name of the image and get its label 
        labels.append(label)                                                # add the label to the labels list
    return labels                                                           # Returning labels of the images 
train_labels = get_labels(train_images, df)                                 # get the lables of the train images 
validation_labels = get_labels(validation_images, df)                       # get the lables of the validation  images 
test_labels = get_labels(test_images, df)                                   # get the lables of the test images 
# Applying transform images
def preprocess_images(image_paths, labels, transform):                      # function for applyig data augmentation on the images 
    images = []  # list for images 
    processed_labels = []  # list for labels of the images 
    for idx, image_path in enumerate(image_paths):                          # Looping through the images 
        image = Image.open(image_path).convert("RGB")                       # Convert to RGB  
        image = transform(image)                                            # Apply transformations  
        images.append(image)                                                # add transformated image to the images list     
        processed_labels.append(labels[idx])                                # add the label of the transformated image to labels list
    return torch.stack(images), torch.tensor(processed_labels)              # Returning transforanted images and thier labels 
# Preprocess the datasets
train_images_transformed, train_labels_transformed = preprocess_images(train_images, train_labels, train_transforms)                         # apply transformation to train images 
validation_images_transformed, validation_labels_transformed = preprocess_images(validation_images, validation_labels, test_transforms)      # apply transformation to validation images 
test_images_transformed, test_labels_transformed = preprocess_images(test_images, test_labels, test_transforms)                              # apply transformation to test images 
# Create DataLoaders
train_dataset = torch.utils.data.TensorDataset(train_images_transformed, train_labels_transformed)                      # create tensor dataset for train images and thier lables 
validation_dataset = torch.utils.data.TensorDataset(validation_images_transformed, validation_labels_transformed)       # create tensor dataset for train images and thier lables 
test_dataset = torch.utils.data.TensorDataset(test_images_transformed, test_labels_transformed)                         # create tensor dataset for train images and thier lables 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)                   # DataLoader for the training dataset
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)        # DataLoader for the validation dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)                    # DataLoader for the test dataset
#Test the DataLoader
for images, labels in train_loader:                  # Looping through a train loader 
    st.write(f"Batch of images shape: {images.shape}")  # Print the shape of the images of the batch 
    st.write(f"Batch of labels shape: {labels.shape}")  # Printing the shape of the labels of the images 
    break                                            # exit after the first batch 
#--------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown("### model archeticture")
st.write(model)
st.markdown('---')
st.markdown("### model training:")
train_losses = []
train_accuracies = []
validation_accuracies = []
validation_losses = []
weights = torch.tensor([1.0, 2.0])
loss_module = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0   
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_module(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)
            loss = loss_module(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.size(0)
    val_accuracy = correct_predictions / total_predictions
    validation_losses.append(val_loss / len(validation_loader))
    validation_accuracies.append(val_accuracy)
    # Print Metrics
    st.write(f"Epoch {epoch+1}/{num_epochs}")
    st.write(f"Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")  
#--------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown("### model Validation:")
model.eval()
val_loss = 0.0
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
    for data, target in validation_loader:  # Validation data loader
        output = model(data)
        loss = loss_module(output, target)
        val_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == target).sum().item()
        total_predictions += target.size(0)
val_accuracy = correct_predictions / total_predictions
# Print accuracy as a percentage
st.write(f"Accuracy of Validation: {val_accuracy * 100:.2f}%")
#--------------------------------------------------------------------------------------------------------
st.markdown('---')
st.markdown("### model testing:")
model.eval()  # Set model to evaluation mode
all_predictions = []
all_labels = []
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
    for images, labels in test_loader:  
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        all_predictions.extend(predicted)
        all_labels.extend(labels)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)    
    test_accuracy = correct_predictions / total_predictions
    st.write(f"Test Accuracy: {test_accuracy:.4f}")
#--------------------------------------------------------------------------------------------------------
# Plotting the loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Loss Curves
ax1.plot(range(1, num_epochs+1), train_losses, label="Training Loss", color='blue', marker='o')
ax1.plot(range(1, num_epochs+1), validation_losses, label="Validation Loss", color='red', marker='x')
ax1.set_title("Loss Curves", fontsize=16)
ax1.set_xlabel("Epoch", fontsize=14)
ax1.set_ylabel("Loss", fontsize=14)
ax1.legend()
ax1.grid(True)
# Accuracy Curves
ax2.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy", color="green")
ax2.plot(range(1, num_epochs + 1), validation_accuracies, label="Validation Accuracy", color="red")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy Curves")
ax2.legend()
st.pyplot(fig)
# Compute confusion matrix
#--------------------------------------------------------------------------------------------------------
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Glaucoma"])
#--------------------------------------------------------------------------------------------------------
# Plot confusion matrix
st.header("Confusion Matrix")
st.write("This confusion matrix visualizes the performance")
fig, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax)
st.pyplot(fig)
# Function to evaluate metrics
def evaluate_metrics(y_true, y_pred_probs, y_pred_labels):
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = np.array(y_pred_labels)   
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, y_pred_probs)   
    # Chi-square statistic
    #chi_square = np.sum(((y_true - y_pred_probs) ** 2) / (y_true + 1e-10))  # Avoid division by zero    
    return accuracy, precision, recall, f1, auc
# Function to evaluate the model on a DataLoader
def evaluate_model(dataloader):
    model.eval()
    all_labels = []
    all_predictions = []
    all_pred_probs = []    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            
            # Convert logits to probabilities (softmax output)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability of class 1
            predictions = (probabilities > 0.5).astype(int)  # Convert probabilities to binary class labels (0 or 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)  # Predicted class labels
            all_pred_probs.extend(probabilities)  # Predicted probabilities

    # Evaluate metrics
    accuracy, precision, recall, f1, auc = evaluate_metrics(np.array(all_labels), np.array(all_pred_probs), np.array(all_predictions))
    return accuracy, precision, recall, f1, auc 

# Function to visualize the metrics
def visualize_metrics(metrics_dict, dataset_name):
    """
    Visualize evaluation metrics.
    Args:
        metrics_dict (dict): Dictionary containing metric names and values.
        dataset_name (str): Name of the dataset (e.g., "Validation", "Test").
    """
    # Create bar plot
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())

    st.bar_chart(metric_names, metric_values, title = f"{dataset_name} Metrics")
    st.write("X-axis: Metric Name")
    st.write("Y-axis: Metric Value")

# Evaluate on validation set
val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate_model(validation_loader)
validation_metrics = {
    "Accuracy": val_accuracy,
    "Precision": val_precision,
    "Recall": val_recall,
    "F1-Score": val_f1,
    "AUC": val_auc
}

# Print validation metrics
st.write("Validation Metrics:")
st.write(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}, AUC: {val_auc:.4f}")

# Visualize validation metrics
visualize_metrics(validation_metrics, "Validation")

# Evaluate on test set
test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate_model(test_loader)
test_metrics = {
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1-Score": test_f1,
    "AUC": test_auc,
    
}

# Print test metrics
st.write("\nTest Metrics:")
st.write(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}, AUC: {test_auc:.4f}")

# Visualize test metrics
visualize_metrics(test_metrics, "Test")
#---------------------------------------------------
st.markdown("---")
st.write("Upload an eye scan image to predict if glaucoma is present.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Transform the image
    input_tensor = test_transforms(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100
        label = "Glaucoma" if predicted.item() == 1 else "No Glaucoma"

        st.write("The prediction is :",label)