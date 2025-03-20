import torch
import os
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from utils.visualization import *

# passiamo alla funzione per il training:  current epoch, modello (CustomNet), dataloader (train_loader), loss (criterion), optimizer
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo... forward the input, find uotput and update parameters
        #compute prediction and loss
        #inputs in ogni riga contine un'immagine
        outputs = model(inputs) # output = matrice con colonne le coppie (classe i- position 0, prob classe i-position 1) per ogni classe
        loss = criterion(outputs, targets)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        running_loss += loss.item() # loss totale dell'epoca corrente
        _, predicted = outputs.max(1) # take most likely class
        total += targets.size(0) # targets.size = dimensione del batch corrente (32) e total accumula numero di processed images
        correct += predicted.eq(targets).sum().item()  # sum of correctly predicted images (sum of predicted = target)

    train_loss = running_loss / len(train_loader) # mean loss over the whole training set
    train_accuracy = 100. * correct / total # accuracy over the whole training set
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')



if __name__ == '__main__':

    #LAB 1

    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform_1()['train'])
    tiny_imagenet_dataset_test = ImageFolder(root='./dataset/tiny-imagenet-200/test', transform=transform_1()['val'])

    # create dataloader
    dataloader_train, _ = dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_test, 64, True, True)

    # Determine the number of classes and samples
    num_classes = len(tiny_imagenet_dataset_train.classes)
    num_samples = len(tiny_imagenet_dataset_train)

    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')

    visualization(dataloader_train)

    # LAB 2
    modify_dataset()
    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform_2())
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform_2())

    # create dataloader
    train_loader, val_loader = dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_val, 32, True, False)

    print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
    print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")


    
