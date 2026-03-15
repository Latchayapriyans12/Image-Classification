# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

## STEP 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

## STEP 2:
Load the FashionMNIST dataset and apply transformations (normalization, tensor conversion).

## STEP 3:
Split the dataset into training and testing sets.

## STEP 4:
Define the CNN architecture with convolutional, pooling, and fully connected layers.

## STEP 5:
Specify the loss function (CrossEntropyLoss) and optimizer (Adam).

## STEP 6:
Train the model using forward pass, loss computation, backpropagation, and parameter updates.

## STEP 7:
Evaluate the model on the test dataset and calculate accuracy.

## STEP 8:
Test the trained model on new/unseen FashionMNIST images.

## PROGRAM

### Name: latchaya priyan S
### Register Number:212224230139
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64*7*7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return x
     

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name:latchaya priyan S')
print('Register Number:212224230139')
summary(model, input_size=(1, 28, 28))
     model = CNNClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
     

## Step 3: Train the Model

def train_model(model, train_loader, num_epochs=3):

    model.train()
    print('Name:latchaya priyan S')
    print('Register Number:212224230139')

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
     

```

## OUTPUT
### Training Loss per Epoch
![alt text](<Screenshot 2026-03-15 222652.png>)


### Confusion Matrix

![alt text](<Screenshot 2026-03-15 222825.png>)

### Classification Report

![alt text](<Screenshot 2026-03-15 222924.png>)

### New Sample Data Prediction

![alt text](<Screenshot 2026-03-15 222930.png>)

## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
