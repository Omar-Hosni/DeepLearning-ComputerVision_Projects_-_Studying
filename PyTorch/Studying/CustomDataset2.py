import torch
from torch import nn
import random
from PIL import Image
import requests
import zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#setting up paths

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

random.seed(42)
image_path_list = list(image_path.glob('*/*/*.jpg'))

random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem

img = Image.open(random_image_path)

#Transforming data
'''
Now what if we wanted to load our image data into PyTorch?

Before we can use our image data with PyTorch we need to:

    Turn it into tensors (numerical representations of our images).
    Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.DataLoader, 
    we'll call these Dataset and DataLoader for short.

'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5), #p=probability of flip
    transforms.ToTensor() #this also converts all pixels values from 0 to 255
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)

    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(f)
            ax[0].set_title(f'original \nsize: {f.size}')
            ax[0].axis('off')

            transformed_images = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_images)
            ax[1].set_title(f'Transformed \nSize: {transformed_images.shape}')
            ax[1].axis('off')

            fig.suptitle(f'class: {image_path.parent.stem}', fontsize=16)
            #plt.show()

plot_transformed_images(image_path_list, transform=data_transform, n=3)


#loading image

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None) # transforms to perform on labels (if necessary)


test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform,
                                 )

class_names = train_data.classes # {'pizza': 0, 'steak': 1, 'sushi': 2}

'''
num_workers defines how many subprocesses 
will be created to load your data.
the more, the more compute power pytorch will use to load your data
'''

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)

from typing import Tuple, Dict, List
import os

'''
Let's write a helper function capable of creating 
a list of class names and a dictionary of 
class names and their indexes given a directory path.
'''
target_directory = train_dir
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path/'train'))])

#print(f'class names found: {class_names_found}')

#make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    #returns (classnames:idx) tuple
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f'coule not find any classes in {directory}')

    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i,cls_name in enumerate(classes)}
    return classes, class_to_idx

#print(find_classes(train_dir))

#create a custom Dataset to replicate ImageFolder
from torch.utils.data import Dataset

class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir, transform=None)->None:
        self.paths = list(Path(targ_dir).glob('*/*.jpg'))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)


    def load_image(self, index) -> Image.Image:
        image_path = self.path[index]
        return Image.open(image_path)

    def __len__(self)->int:
        return len(self.paths)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        #returns one sample of data, data and label (X,y)
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]


        #return data, label(X,y)
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx





# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom created train Dataset
                                     batch_size=1, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=False) # don't usually need to shuffle testing data


'''
another suggested way of augmenting

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense 
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])
'''

# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# Create testing transform (no data augmentation)
test_transform_2 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

#lets build model

# Setup batch size and number of workers
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
#print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")


# Turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform_2)


# Create DataLoader's
train_dataloader_simple = DataLoader(train_data_augmented,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=output_shape)
        )

    def forward(self,x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


model = TinyVGG(3,10,len(train_data.classes)).to(device)

#train and test

def train_step(nn_model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss, train_acc = 0,0

    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        y_pred = nn_model(X)

        loss = loss_fn(y, y_pred)

        optimizer.zero_grad()

        loss.backwards()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(nn_model,dataloader,loss_fn):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = nn_model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc



from tqdm.auto import tqdm

def train(nn_model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs:int=5):

    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_accc': [],
    }

    for i in tqdm(range(epochs)):

        train_loss, train_acc = train_step(nn_model, train_dataloader, loss_fn, optimizer)
        test_loss, test_acc = test_step(nn_model, train_dataloader, loss_fn, optimizer)


        print(
            f"Epoch: {i + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


if __name__ == '__main__':

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_epoch = 5

    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=(len(train_data_augmented.classes))).to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters() ,lr=0.001)

    from timeit import default_timer as timer
    start_time = timer()

    model_res = train(model, train_dataloader_simple, test_dataloader_simple, loss_fn=loss_fn, optimizer=optimizer, epochs=num_epoch)

    end_time = timer()

    print(f"Total training time: {end_time - start_time:.3f} seconds")


