#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import warnings
import copy
warnings.filterwarnings("ignore")

cross_validation_number = sys.argv[1]
path_prefix_algorithm = sys.argv[2]
withBaselineMetabolome = sys.argv[3]
print(cross_validation_number, path_prefix_data, withBaselineMetabolome)
path_prefix_data = "/".join(path_prefix_algorithm.split("/")[:-2]) + "/processed_data/"
if os.path.exists(path_prefix_algorithm)==False:
    os.mkdir(path_prefix_algorithm)

# Load Data
X_train_ori = pd.read_csv(path_prefix_data + "X_train.csv", header=None).values;
y_train_ori = pd.read_csv(path_prefix_data + "y_train.csv", header=None).values;
X_test_ori = pd.read_csv(path_prefix_data + "X_test.csv", header=None).values;
y_test_ori = pd.read_csv(path_prefix_data + "y_test.csv", header=None).values;
compound_name = pd.read_csv(path_prefix_data + "compound_names.csv", delimiter="\t", header=None).values.flatten();
Nm = compound_name.shape[0]
Ni = X_train_ori.shape[1] - y_train_ori.shape[1] #### interventions
#print(X_train_ori.shape, y_train_ori.shape, X_validation_ori.shape, y_validation_ori.shape, Nm, Ni)

# re-split the data again
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
X_ori = np.concatenate([X_train_ori, X_test_ori], axis=0)
y_ori = np.concatenate([y_train_ori, y_test_ori], axis=0)
X_train_ori, X_test_ori, y_train_ori, y_test_ori = train_test_split(X_ori, y_ori, test_size=0.2, random_state=i_split+42)
X_train_ori, X_validation_ori, y_train_ori, y_validation_ori = train_test_split(X_train_ori, y_train_ori, test_size=0.2, random_state=42)

# ### Hyperparameter selection
#### import libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout, Nlayers):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
            # Define proportion or neurons to dropout
            self.dropout = torch.nn.Dropout(dropout)
            self.Nlayers = Nlayers
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            relu_dropout = self.dropout(relu)
            for i in range(self.Nlayers-1):
                hidden = self.fc2(relu_dropout)
                relu = self.relu(hidden)
                relu_dropout = self.dropout(relu)
            output = self.fc3(relu_dropout)
            return output

def spearman(target, pred):
    pred = pred.detach().numpy()
    target = target.detach().numpy()
    df1 = pd.DataFrame(pred)
    df2 = pd.DataFrame(target)
    #metabolites_corr = df1.corrwith(df2, axis = 0, method='pearson').values
    metabolites_corr = df1.corrwith(df2, axis = 0, method='spearman').values
    return np.nanmean(metabolites_corr)

def prediction(X_train, X_validation, X_test, y_train, y_validation, y_test, Nlayers, layer_size, weight_decay, dropout):
    #### convert to tensor
    X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
    X_validation = torch.FloatTensor(X_validation); y_validation = torch.FloatTensor(y_validation)
    X_test = torch.FloatTensor(X_test); y_test = torch.FloatTensor(y_test)
    
    # Use torch.utils.data to create a DataLoader 
    # that will take care of creating batches 
    BATCH_SIZE = 64
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    #### Train the model
    model = Feedforward(X_train.shape[1], layer_size, y_train.shape[1], dropout, Nlayers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)

    #### Train the model
    model.eval()
    y_pred = model(X_validation)
    before_train = spearman(y_pred.squeeze(), y_validation) 
    model.train()
    epoch = 1000
    validation_SCC = []
    model_list = []
    for epoch in range(epoch):
        # Loop over batches in an epoch using DataLoader
        model.train()
        for id_batch, (X_batch, y_batch) in enumerate(dataloader):
            # Forward pass
            y_batch_pred = model(X_batch)
            # Compute Loss
            loss = criterion(y_batch_pred.squeeze(), y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        y_pred = model(X_validation)
        after_train = spearman(y_pred.squeeze(), y_validation) 
        validation_SCC = validation_SCC + [after_train]
        model.eval()
        y_pred = model(X_test)
        test_SCC = spearman(y_pred.squeeze(), y_test)
        model_list = model_list + [copy.deepcopy(model)]
        if (loss < 0.1) and (validation_SCC[-1] - validation_SCC[-21]) < 0: 
            model = copy.deepcopy(model_list[-(20-np.argmax(validation_SCC[-20:]))]) 
            break

    #### Evaluate the model
    model.eval()
    y_pred = model(X_test)
    after_train = spearman(y_pred.squeeze(), y_test) 
    return model, y_pred



# ### 5-fold cross-validation of predicting microbiome compostion at time t+1
if withBaselineMetabolome==True:
    X_train = X_train_ori.copy()
    X_validation = X_validation_ori.copy()
    X_test = X_test_ori.copy()
    y_train = y_train_ori[:, :-Nm]
    y_validation = y_validation_ori[:, :-Nm]
    y_test = y_test_ori[:, :-Nm]
else:
    X_train = X_train_ori.copy()
    X_validation = X_validation_ori.copy()
    X_test = X_test_ori.copy()
    X_train = np.concatenate([X_train[:, :-Nm-Ni], X_train[:, -Ni:]], axis=1)
    X_validation = np.concatenate([X_validation[:, :-Nm-Ni], X_validation[:, -Ni:]], axis=1)
    X_test = np.concatenate([X_test[:, :-Nm-Ni], X_test[:, -Ni:]], axis=1)
    y_train = y_train_ori[:, :-Nm]
    y_validation = y_validation_ori[:, :-Nm]
    y_test = y_test_ori[:, :-Nm]

Nlayers = 6
layer_size = 2048
weight_decay = 0
dropout = 0.0


from sklearn.preprocessing import StandardScaler 
scaler_x_1 = StandardScaler()  
scaler_x_1.fit(X_train)  
X_train = scaler_x_1.transform(X_train)  
X_validation = scaler_x_1.transform(X_validation)  
X_test = scaler_x_1.transform(X_test)  
scaler_y_1 = StandardScaler()  
scaler_y_1.fit(y_train)
y_train = scaler_y_1.transform(y_train)  
y_validation = scaler_y_1.transform(y_validation)
y_test = scaler_y_1.transform(y_test) 

#### convert to tensor
X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
X_validation = torch.FloatTensor(X_validation); y_validation = torch.FloatTensor(y_validation)
X_test = torch.FloatTensor(X_test); y_test = torch.FloatTensor(y_test)

model, y_pred = prediction(X_train, X_validation, X_test, y_train, y_validation, y_test, Nlayers, layer_size, weight_decay, dropout)

import copy
model_1 = copy.deepcopy(model)
X_train_1 = copy.deepcopy(X_train)
X_validation_1 = copy.deepcopy(X_validation)
X_test_1 = copy.deepcopy(X_test)
y_train_1 = copy.deepcopy(y_train)
y_validation_1 = copy.deepcopy(y_validation)
y_test_1 = copy.deepcopy(y_test)


# ### 5-fold cross-validation of predicting metabolome at time t+1 based on microbiome compostion at time t+1
if withBaselineMetabolome==True:
    X_train = np.concatenate([scaler_y_1.inverse_transform(model_1(X_train_1).detach().numpy()), X_train_ori[:, -Nm-Ni:]], axis=1)
    X_validation = np.concatenate([scaler_y_1.inverse_transform(model_1(X_validation_1).detach().numpy()), X_validation_ori[:, -Nm-Ni:]], axis=1)
    X_test = np.concatenate([scaler_y_1.inverse_transform(model_1(X_test_1).detach().numpy()), X_test_ori[:, -Nm-Ni:]], axis=1)
    y_train = y_train_ori[:, -Nm:]
    y_validation = y_validation_ori[:, -Nm:]
    y_test = y_test_ori[:, -Nm:]
else:
    X_train = np.concatenate([scaler_y_1.inverse_transform(model_1(X_train_1).detach().numpy()), X_train_ori[:, -Ni:]], axis=1)
    X_validation = np.concatenate([scaler_y_1.inverse_transform(model_1(X_validation_1).detach().numpy()), X_validation_ori[:, -Ni:]], axis=1)
    X_test = np.concatenate([scaler_y_1.inverse_transform(model_1(X_test_1).detach().numpy()), X_test_ori[:, -Ni:]], axis=1)
    y_train = y_train_ori[:, -Nm:]
    y_validation = y_validation_ori[:, -Nm:]
    y_test = y_test_ori[:, -Nm:]

Nlayers = 6
layer_size = 2048
weight_decay = 0
dropout = 0.0

from sklearn.preprocessing import StandardScaler 
scaler_x_2 = StandardScaler()  
scaler_x_2.fit(X_train)  
X_train = scaler_x_2.transform(X_train)  
X_validation = scaler_x_2.transform(X_validation)  
X_test = scaler_x_2.transform(X_test)  
scaler_y_2 = StandardScaler()  
scaler_y_2.fit(y_train)
y_train = scaler_y_2.transform(y_train)  
y_validation = scaler_y_2.transform(y_validation)  
y_test = scaler_y_2.transform(y_test)  

#### convert to tensor
X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
X_validation = torch.FloatTensor(X_validation); y_validation = torch.FloatTensor(y_validation)
X_test = torch.FloatTensor(X_test); y_test = torch.FloatTensor(y_test)

model, y_pred = prediction(X_train, X_validation, X_test, y_train, y_validation, y_test, Nlayers, layer_size, weight_decay, dropout)

import copy
model_2 = copy.deepcopy(model)
X_train_2 = copy.deepcopy(X_train)
X_validation_2 = copy.deepcopy(X_validation)
X_test_2 = copy.deepcopy(X_test)
y_train_2 = copy.deepcopy(y_train)
y_validation_2 = copy.deepcopy(y_validation)
y_test_2 = copy.deepcopy(y_test)


from scipy.stats import spearmanr
x_plot = y_pred.detach().numpy()
y_plot = y_test.detach().numpy()
metabolites_corr = np.zeros(Nm)
for i in range(Nm):
    metabolites_corr[i] = spearmanr(x_plot[:,i+x_plot.shape[1]-Nm], y_plot[:,i+x_plot.shape[1]-Nm])[0]

metabolites_corr_annotated = metabolites_corr[compound_name==compound_name]
compound_name_annotated = compound_name[compound_name==compound_name]

print("="*50)
print("The mean Spearman C.C. for all metabolites is")
print(np.nanmean(metabolites_corr))

########### Save data
if withBaselineMetabolome==True:
    np.savetxt("./results/true_metabolomic_profiles_wb_"+str(cross_validation_number)+".csv", y_test.detach().numpy(), delimiter=',')
    np.savetxt("./results/predicted_metabolomic_profiles_wb_"+str(cross_validation_number)+".csv", y_pred.detach().numpy(), delimiter=',')
    np.savetxt("./results/metabolites_corr_wb_"+str(cross_validation_number)+".csv", metabolites_corr, delimiter=',')
else:
    np.savetxt("./results/true_metabolomic_profiles_wob_"+str(cross_validation_number)+".csv", y_test.detach().numpy(), delimiter=',')
    np.savetxt("./results/predicted_metabolomic_profiles_wob_"+str(cross_validation_number)+".csv", y_pred.detach().numpy(), delimiter=',')
    np.savetxt("./results/metabolites_corr_wob_"+str(cross_validation_number)+".csv", metabolites_corr, delimiter=',')
    

