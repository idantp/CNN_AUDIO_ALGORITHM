from gcommand_loader import GCommandLoader
import torch
from torch import nn, optim
import torch.utils.data
import torch.nn.functional as F
import ConvolutionClass as Cnn
import numpy as np

LEARNING_RATE = 0.001

EPOCHS = 16


def test_conv(cnn_model, test_loader):
    cnn_model.eval()
    with torch.no_grad():
        correct_predict = 0
        total = 0
        for images, labels in test_loader:
            # images = images.to("cuda")
            # labels = labels.to("cuda")
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_predict += (predicted == labels).sum().item()
        print('cnn_model accuracy: {} %'.format((correct_predict / total) * 100))


def train_conv(training_set, optimizer, epochs, criterion, cnn_model):
    accuracy_list = []
    cnn_model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_set):
            # images = images.to("cuda")
            # labels = labels.to("cuda")
            # Run the forward pass
            outputs = cnn_model(images)
            loss = criterion(outputs, labels)

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Accuracy: {:.2f}%'.format(epoch + 1, EPOCHS, (correct / total) * 100))


def get_predictions(testing_set, loader, model):
    model.eval()
    prediction_list = []
    file_names = []

    with torch.no_grad():
        for images, labels in testing_set:
            # images = images.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for p in predicted:
                prediction_list.append(p.item())

    for i, data in enumerate(loader.spects):
        name = data[0].split('/')[5]
        file_names.append(name)
        
    return file_names, prediction_list


def get_inputs(cnn_model):
    training = GCommandLoader('./ML4_dataset/data/train')

    training_set = torch.utils.data.DataLoader(training, batch_size=100, shuffle=True, num_workers=20, pin_memory=True,
                                               sampler=None)
    validation = GCommandLoader('./ML4_dataset/data/valid')
    validation_set = torch.utils.data.DataLoader(validation, batch_size=100, shuffle=False, num_workers=20,
                                                 pin_memory=True, sampler=None)
    testing = GCommandLoader('./ML4_dataset/data/test1')
    testing_set = torch.utils.data.DataLoader(testing, batch_size=100, shuffle=False, num_workers=20, pin_memory=True,
                                              sampler=None)

    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    return training_set, validation_set, testing_set, optimizer


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = Cnn.ConvolutionNN()#.to(device)
    training_set, validation_set, testing_set, optimizer = get_inputs(cnn_model)
    criterion = nn.CrossEntropyLoss()
    train_conv(training_set, optimizer, EPOCHS, criterion, cnn_model)
    test_conv(cnn_model, validation_set)
    loader = GCommandLoader('./ML4_dataset/data/test1')
    prediction_list, file_names = get_predictions(testing_set, loader, cnn_model)


if __name__ == '__main__':
    main()
