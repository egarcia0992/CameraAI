import torch
from tensorflow.python.data.ops.dataset_ops import DatasetSpec
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu" #check if CUDA gpu software is available
# print("Device available: ", device)

image_path = []
labels = []
main_directory = r"" # Put the directory with the training data here

# CREATING DATA FRAME FROM FILES
for label in os.listdir(main_directory):
    # print(label)
    for image in os.listdir(main_directory + fr"\{label}"):
        # print(image)
        image_path.append(main_directory + fr"\{label}" + fr"\{image}")
        labels.append(label)

data_df = pd.DataFrame(zip(image_path, labels), columns = ["image_path", "labels"])
# print(data_df["labels"].unique()) # See the possible labels
pd.set_option('display.max_colwidth', None) # Print full image path
print(data_df.head()) # First 5 rows of the dataframe

# SETTING UP TRAIN, TEST, AND VALIDATE DATA
train = data_df.sample(frac=0.7)
test = data_df.drop(train.index)
val = test.sample(frac=0.5)
test = test.drop(val.index)
# print(train.shape)
# print(val.shape)
# print(test.shape)

label_encoder = LabelEncoder()
label_encoder.fit(data_df["labels"])
# print("Label Encoder Classes:", label_encoder.classes_)
# print(data_df['labels'].value_counts()) # Print number of items in each label
# MAKE ALL IMAGES SIMILAR IN SIZE AND TYPE
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform = None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe['labels'])).to(device)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        img_path = self.dataframe.iloc[index, 0]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image).to(device)

        return image, label

train_dataset = CustomImageDataset(dataframe= train, transform= transform)
val_dataset = CustomImageDataset(dataframe= val, transform= transform)
test_dataset = CustomImageDataset(dataframe= test, transform= transform)

# print(train_dataset.__getitem__(2))
# print(label_encoder.inverse_transform([0])) # Checks what label corresponds to the integer 0

# SHOW A FEW IMAGES
# n_rows = 3
# n_columns = 3
# f, axarr = plt.subplots(n_rows, n_columns)
#
# for row in range(n_rows):
#     for column in range(n_columns):
#         image = Image.open(data_df.sample(n = 1)["image_path"].iloc[0]).convert("RGB")
#         axarr[row, column].imshow(image)
#         axarr[row, column].axis('off')
# plt.show()

LR = 3e-4
BATCH_SIZE = 16
EPOCHS = 10

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 128)

        self.output = nn.Linear(128, len(data_df['labels'].unique()))

    def forward(self, x):
        x = self.conv1(x) # -> (32, 128, 128)
        x = self.pooling(x) # -> (32, 64, 64)
        x = self.relu(x)

        x = self.conv2(x) # -> (64, 64, 64)
        x = self.pooling(x) # -> (64, 32, 32)
        x = self.relu(x)

        x = self.conv3(x) # -> (128, 32, 32)
        x = self.pooling(x) # -> (128, 16, 16)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x

model = Net().to(device)

# from torchsummary import summary
# summary(model, input_size = (3, 128, 128))

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LR)

total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train= 0
    total_loss_val = 0
    total_acc_val = 0

    for inputs, labels, in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        total_loss_train += train_loss.item()

        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()
    with torch.no_grad():
        for inputs, labels, in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            total_loss_val += val_loss.item()

            val_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
            total_acc_val += val_acc

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_validation_plot.append(round(total_loss_val/1000, 4))

    total_acc_train_plot.append(round(total_acc_train/train_dataset.__len__() * 100, 4))
    total_acc_validation_plot.append(round(total_acc_val/val_dataset.__len__() * 100, 4))

    print(f'''Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/1000, 4)},
    Train Accuracy: {round(total_acc_train/train_dataset.__len__() * 100, 4)}
    Validation Loss: {round(total_loss_val/1000, 4)},
    Validation Accuracy: {round(total_acc_val/val_dataset.__len__() * 100, 4)}
''')

with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for inputs, labels in test_loader:
        predictions = model(inputs)

        acc = (torch.argmax(predictions, axis = 1) == labels).sum().item()
        total_acc_test += acc
        test_loss = criterion(predictions, labels)
        total_loss_test += test_loss.item()

print(f"Accuracy score is: {round(total_acc_test/test_dataset.__len__() * 100, 4)} and loss is: {round(total_loss_test/1000, 4)}")
# print(acc)

# fig, axs = plt.subplots(nrows =1, ncols = 2, figsize = (15, 5))
# axs[0].plot(total_loss_train_plot, label = 'Training Loss')
# axs[0].plot(total_loss_validation_plot, label = 'Validation Loss')
# axs[0].set_title('Training and Validation Loss over Epochs')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].legend()
# axs[1].plot(total_acc_train_plot, label = 'Training Accuracy')
# axs[1].plot(total_acc_validation_plot, label = 'Validation Accuracy')
# axs[1].set_title('Training and Validation Accuracy over Epochs')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('Accuracy')
# axs[1].legend()
# plt.show()

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).to(device)
    output = model(image.unsqueeze(0))
    print(output)
    output = torch.argmax(output, axis = 1).item()
    return label_encoder.inverse_transform([output])

# Test with a known images
masked_img = data_df[data_df['labels'] == 'with_mask'].iloc[0]['image_path']
print("Prediction for masked image:", predict_image(masked_img))  # Should output 'with_mask'

unmasked_img = data_df[data_df['labels'] == 'without_mask'].iloc[0]['image_path']
print("Prediction for unmasked image:", predict_image(unmasked_img))  # Should output 'without_mask'
