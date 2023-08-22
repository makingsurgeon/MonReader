import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from itertools import count
import natsort
import numpy as np
import os
import math
import ssl

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy
import random
import pandas as pd
import tqdm
torch.manual_seed(10)

#Using Aplle M1 to run the algorithm, it is a transfer learning method based on Effinet
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

class Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=False):
        super(Dataset, self).__init__()
        self.image_paths = image_paths
        self.labels = labels    #.astype(dtype='int')
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    

def get_transform(model_name):
    transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])  
    
    return transform

train_image_paths = []
basepath = os.path.dirname(__file__)
filename = path.abspath(path.join(basepath, "..","data/images/training/notflip"))
train_image_paths.append(glob.glob(filename + '/*'))
filename1 = path.abspath(path.join(basepath, "..","data/images/training/flip"))
train_image_paths.append(glob.glob(filename1 + '/*'))
train_image_paths1 = [item for sublist in train_image_paths for item in sublist]
train_image_paths1 = natsort.natsorted(train_image_paths1)

l = []
for i in range(1230):
    l.append(0)
for i in range(1162):
    l.append(1)

percentile_list = pd.DataFrame(
    {'Link': train_image_paths1,
     'Label': l,
    })

percentile_list1 = percentile_list.sample(frac=1, random_state=1)
train_image_paths = percentile_list1.loc[:,"Link"].to_numpy().tolist()
labels = percentile_list1.loc[:,"Label"].to_numpy().tolist()
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):]  
train_labels, valid_labels = labels[:int(0.8*len(labels))], labels[int(0.8*len(labels)):] 

summary = {i:0 for i in range(2)}
num_classes = 2
total_samples = 0
for i in train_labels:
    total_samples += 1
    summary[i] += 1

class Classifier():

    def __init__(self, name, model, dataloaders, parameter, use_cuda=False):
        
        '''
        @name: Experiment name. Will define stored results etc. 
        @model: Any models
        @dataloaders: Dictionary with keys train, val and test and corresponding dataloaders
        @class_names: list of classes, where the idx of class name corresponds to the label used for it in the data
        @use_cuda: whether or not to use cuda
        '''
       
        self.name = name
        if use_cuda and not torch.backends.mps.is_available():
            raise Exception("Asked for MPS but not found")
            
        self.use_cuda = use_cuda
        self.epoch = parameter['epochs']
        self.lr = parameter['lr']
        self.batch_size = parameter['batch_size']
        
        self.model = model.to('mps' if use_cuda else 'cpu') # model.to('cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loader, self.valid_loader = self.get_dataloaders(dataloaders['train_image_paths'], 
                                                                    dataloaders['train_labels'], 
                                                                    dataloaders['valid_image_paths'], 
                                                                    dataloaders['valid_labels'], 
                                                                    train_transforms=dataloaders['transforms'], 
                                                                    batch_size = self.batch_size,
                                                                    shuffle=parameter['shuffle'],
                                                                   )
        self.class_names = parameter['class_names']
        
        self.activations_path = os.path.join('activations', self.name)
        self.kernel_path = os.path.join('kernel_viz', self.name)
        save_path = "/Users/zihuiouyang/Downloads/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not os.path.exists(self.activations_path):
            os.makedirs(self.activations_path)

        if not os.path.exists(self.kernel_path):
            os.makedirs(self.kernel_path)
            
        self.save_path = save_path

    def train(self, save=True):
        '''
        @epochs: number of epochs to train
        @save: whether or not to save the checkpoints
        '''
        best_val_accuracy = - math.inf
        
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            self.model.train()
            t = time.time()
            running_loss = 0.0
            train_acc = 0
            val_accuracy = 0
            correct = 0
            total = 0
            count = 0
            loop = tqdm.tqdm(self.train_loader, total = len(self.train_loader), leave = True)
            
            for img, label in loop:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = img.to(device), label.to(device) #img.to(device), label.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                total += labels.shape[0]
                correct += (predictions == labels).sum().item()
                
                count += 1
                if count % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {count + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
                    
            train_acc = 100 * correct / total
            print(f'Epoch:', epoch + 1, f'Training Epoch Accuracy:{train_acc}')
            
            # evaluate the validation dataset
            self.model.eval()
            correct_pred = {classname: 0 for classname in self.class_names}
            total_pred = {classname: 0 for classname in self.class_names}

            # again no gradients needed
            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.valid_loader:
                    images, labels = data[0].to(device), data[1].to(device) #data[0], data[1]
                    outputs = self.model(images)
                    _, predictions = torch.max(outputs, 1)
                    # collect the correct predictions for each class
                    total += labels.shape[0]
                    correct += (predictions == labels).sum().item()

                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1

            val_accuracy = 100 * correct / total
            print(f'Epoch:', epoch + 1, f'Validation Epoch Accuracy:{val_accuracy}')
                        
            # print the summary for each class
            print('Epoch:', epoch + 1, 'Correct predictions', correct_pred)
            print('Epoch:', epoch + 1, 'Total predictions', total_pred)
            print('Epoch:', epoch + 1, 'Correct predictions', correct_pred)
            print('Epoch:', epoch + 1, 'Total predictions', total_pred)
            
            # inspect the time taken to train one epoch
            d = time.time()-t
            print('Fininsh Trainig Epoch', epoch, '!', 'Time used:', d)
            
            if save:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model4.pt"))
                if val_accuracy > best_val_accuracy:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                    best_val_accuracy = val_accuracy

        print('Done training!')                       

    
    def evaluate(self):
        # for evaluating the test dataset if there were any.
        try:
            assert os.path.exists(os.path.join(self.save_path, 'best.pt'))
            
        except:
            print('Please train first')
            return
        
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best.pt')))
        self.model.eval()
        
    def get_dataloaders(self, train_image_paths, train_labels, valid_image_paths, valid_labels, train_transforms=False, batch_size=32, shuffle=True):
        train_dataset = Dataset(train_image_paths,train_labels, train_transforms)
        val_dataset = Dataset(valid_image_paths,valid_labels, train_transforms)
        train_loader = DataLoader(train_dataset, batch_size, shuffle)
        valid_loader = DataLoader(val_dataset, batch_size, shuffle = True)
        return train_loader, valid_loader
    
    def grad_cam_on_input(self, img):
        
        try:
            assert os.path.exists(os.path.join(self.save_path, 'best.pt'))

        except:
            print('It appears you are testing the model without training. Please train first')
            return

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best.pt')))


        self.model.eval()
        img = img.to('mps' if self.use_cuda else 'cpu')


        out = self.model(img)

        _, pred = torch.max(out, 1)

        predicted_class = self.class_names[int(pred)]
        print(f'Predicted class was {predicted_class}')

        out[:, pred].backward()
        gradients = self.model.get_gradient_activations()

        print('Gradients shape: ', f'{gradients.shape}')

        mean_gradients = torch.mean(gradients, [0, 2, 3]).cpu()
        activations = self.model.get_final_conv_layer(img).detach().cpu()

        print('Activations shape: ', f'{activations.shape}')

        for idx in range(activations.shape[1]):
            activations[:, idx, :, :] *= mean_gradients[idx]

        final_heatmap = np.maximum(torch.mean(activations, dim=1).squeeze(), 0)

        final_heatmap /= torch.max(final_heatmap)

        return final_heatmap

    def trained_kernel_viz(self):
        
        all_layers = [0, 3]
        all_filters = []
        for layer in all_layers:

            filters = self.model.conv_model[layer].weight
            all_filters.append(filters.detach().cpu().clone()[:8, :8, :, :])

        for filter_idx in range(len(all_filters)):

            filter = all_filters[filter_idx]
            print(filter.shape)
            filter = filter.contiguous().view(-1, 1, filter.shape[2], filter.shape[3])
            image = show_img(make_grid(filter))
            image = 255 * image
            cv2.imwrite(os.path.join(self.kernel_path, f'filter_layer{all_layers[filter_idx]}.jpg'), image)
    

    def activations_on_input(self, img):
        
        img = img.to('mps' if self.use_cuda else 'cpu')

        all_layers = [0,3,6,8,10]
        all_viz = []
        
        # looking at the outputs of the relu
        for each in all_layers:

            current_model = self.model.conv_model[:each+1]
            current_out = current_model(img)
            all_viz.append(current_out.detach().cpu().clone()[:, :64, :, :])

        for viz_idx in range(len(all_viz)):

            viz = all_viz[viz_idx]
            viz = viz.view(-1, 1, viz.shape[2], viz.shape[3])
            image = show_img(make_grid(viz))
            image = 255 * image
            cv2.imwrite(os.path.join(self.activations_path, f'sample_layer{all_layers[viz_idx]}.jpg'), image)    
        
ssl._create_default_https_context = ssl._create_unverified_context
example_model = models.efficientnet_b7(pretrained=True)

class TransferEffiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_effi_net = models.efficientnet_b7(pretrained=True)
        self.conv_model = self.get_conv_layers()
        self.avg_pool = self.transition_layer()
        self.fc_model = self.get_fc_layers()
        self.activate_training_layers()

    def activate_training_layers(self):
        for name, param in self.conv_model.named_parameters():
            number = int(name.split('.')[1])
            # for all layers except the last conv layer, set param.requires_grad = False
            if number == 8:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        for name, param in self.fc_model.named_parameters():
            # for all of these layers set param.requires_grad as True
            param.requires_grad = True

    def get_conv_layers(self):
        return self.base_effi_net.features

    def transition_layer(self):
        return self.base_effi_net.avgpool

    def get_fc_layers(self):
        return nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2560, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=2, bias=True),
        )
    
    def forward(self, x):
        x = self.conv_model(x)   #call the conv layers
        x = self.avg_pool(x)  #call the avg pool layer
        x = torch.flatten(x, 1)
        x = self.fc_model(x)  #call fully connected layers  
        
        return x
    
name = 'TransferEffiNet'
classes = [i for i in range(2)]
transforms = get_transform('effinet')
dataloaders = {'train_image_paths': train_image_paths, 'train_labels' : train_labels, 'valid_image_paths': valid_image_paths, 'valid_labels':valid_labels, 'transforms':transforms}
parameters = {'lr': 0.001, 'epochs' : 5, 'batch_size':32, 'shuffle':False, 'class_names':classes}

model = TransferEffiNet()
classifier = Classifier(name, model, dataloaders, parameters, use_cuda=True)
classifier.train()



test_image_paths = []
filename = path.abspath(path.join(basepath, "..","data/images/testing/notflip"))
test_image_paths.append(glob.glob(filename + '/*'))
filename1 = path.abspath(path.join(basepath, "..","data/images/testing/flip"))
test_image_paths.append(glob.glob(filename1 + '/*'))

test_image_paths1 = [item for sublist in test_image_paths for item in sublist]
test_image_paths1 = natsort.natsorted(test_image_paths1)


l1 = []
for i in range(307):
    l1.append(0)
for i in range(290):
    l1.append(1)

def get_pred(model, train_transforms, batch_size, use_cuda=True):
    test_dataset = Dataset(test_image_paths1, l1, train_transforms)
    test_loader = DataLoader(test_dataset, batch_size, shuffle = False)
    model = model.to('mps' if use_cuda else 'cpu')
    pr = []
    pred = []
    l = []
    # again no gradients needed
    t = time.time()
    negative_examples = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            l.append(labels)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            m = F.softmax(outputs, dim=1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                pred.append(prediction)
            for p in m:
                pr.append(p)
                
    processtime = time.time()-t
    print('processtime', processtime)
    return l, pred, pr, processtime

save_path = os.getcwd()
best_effi = TransferEffiNet()
best_effi.load_state_dict(torch.load(os.path.join(save_path, 'best.pt')))
transforms = get_transform('effinet')
batch_size = 32
y_test_true, y_test_predicted, pr, time = get_pred(best_effi, transforms, batch_size)

for i in range(len(y_test_true)):
    y_test_true[i] = y_test_true[i].cpu()
for i in range(len(y_test_true)):
    y_test_true[i] = y_test_true[i].data.numpy()
y_test_true = [item for sublist in y_test_true for item in sublist]
for i in range(len(y_test_true)):
    y_test_predicted[i] = y_test_predicted[i].cpu().data.numpy()