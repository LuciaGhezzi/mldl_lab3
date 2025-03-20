import torch
import os
from torchvision.datasets import ImageFolder
from dataset.transform_dataset import *
from data.dataloader import *
from utils.visualization import *
from models.custom_net import *
from train import *

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad(): # non calcolare i gradienti essendo nella fase di validation: ora 'testiamo' quelli che abbiamo ottimizzaro prima nel training
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # todo...
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # come sopra
            val_loss += loss.item() #accumulates tot validation loss
            _, predicted = outputs.max(1) #finds most likely class and get predictions
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader) # mean loss over the whole validation set
    val_accuracy = 100. * correct / total # accuracy over the whole validation set

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

if __name__ == '__main__':
    # LAB 2
    # Load the dataset
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform_2())
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/test', transform=transform_2())

    # create dataloader
    _, val_loader = dataloader(tiny_imagenet_dataset_train, tiny_imagenet_dataset_val, 32, True, False)

    model = CustomNet().cuda() # crea istanza della nostra rete neurale CostumeNet .cuda = Sposta il modello sulla GPU se disponibile
    criterion = nn.CrossEntropyLoss() # define the entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # crea un ottimizzatore SGD, responsabile dell'aggiornamento dei pesi del modello durante l'addestramento per ridurre al minimo la loss

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        # train the model
        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')
