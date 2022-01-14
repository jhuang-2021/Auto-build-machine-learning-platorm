import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
                              
                              
dwn=False

valset = datasets.MNIST('./data', download=dwn, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
num_of_images = 60

input_size = 784
hidden_sizes = [128, 64]
output_size = 10
modelFn='./trained_model.mdl'

predicts=[]

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

model.load_state_dict( torch.load(modelFn) )
criterion = nn.NLLLoss()

images, labels = next(iter(valloader))

img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
digit=str(probab.index(max(probab)))
print("Predicted Digit =", digit)
predicts.append(digit)

correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1
acc=round(float(correct_count)/float(all_count)*100.0,2)
res="Predicted digits for classification:\n "
res+="    Number of predictions: "+str(all_count)+'\n'
res+="    Number of correct predictions: "+str(correct_count)+'\n'
res+="    Accuracy:                    "+str(acc)+"%\n"
res+="   Predicted digits:\n"
for dig in predicts:
    res+=(dig+'\n')
    
fd=open('result.dat'.'w')

