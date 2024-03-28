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

path_prefix_data = sys.argv[1]
withBaselineMetabolome = sys.argv[2]
print(path_prefix_data, withBaselineMetabolome)

# Load Data
df_X_train_ori = pd.read_csv(path_prefix_data + "X_train.csv", index_col=0)
df_y_train_ori = pd.read_csv(path_prefix_data + "y_train.csv", index_col=0)
df_X_test_ori = pd.read_csv(path_prefix_data + "X_test.csv", index_col=0)
df_y_test_ori = pd.read_csv(path_prefix_data + "y_test.csv", index_col=0)
X_train_ori = df_X_train_ori.values;
y_train_ori = df_y_train_ori.values;
X_test_ori = df_X_test_ori.values;
y_test_ori = df_y_test_ori.values;
compound_name = pd.read_csv(path_prefix_data + "compound_names.csv", delimiter="\t", header=None).values.flatten();
Nm = compound_name.shape[0]
Ni = X_train_ori.shape[1] - y_train_ori.shape[1] #### interventions

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

def MLP(Nlayers, layer_size, weight_decay, dropout, X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import StandardScaler 
    scaler_x = StandardScaler()  
    scaler_x.fit(X_train)
    X_train = scaler_x.transform(X_train)  
    X_test = scaler_x.transform(X_test)  
    scaler_y = StandardScaler()  
    scaler_y.fit(y_train)
    y_train = scaler_y.transform(y_train)  
    y_test = scaler_y.transform(y_test)  
    
    #### convert to tensor
    X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
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
    y_pred = model(X_test)
    before_train = spearman(y_pred.squeeze(), y_test) 
    model.train()
    epoch = 1000
    test_error = []
    for epoch in range(epoch):
        # Loop over batches in an epoch using DataLoader
        for id_batch, (X_batch, y_batch) in enumerate(dataloader):
            #print(id_batch)
            # Forward pass
            y_batch_pred = model(X_batch)
            # Compute Loss
            loss = criterion(y_batch_pred.squeeze(), y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%20==0:
            model.eval()
            y_pred = model(X_test)
            after_train = spearman(y_pred.squeeze(), y_test) 
            test_error = test_error + [after_train]
            if (loss < 0.1) and (test_error[-1] - test_error[-2]) < 0:
                model = copy.deepcopy(previous_model)
                break
            previous_model = copy.deepcopy(model)

    #### Evaluate the model
    model.eval()
    y_pred = model(X_test)
    after_train = spearman(y_pred.squeeze(), y_test) 

    from scipy.stats import spearmanr
    x_plot = y_pred.detach().numpy()
    y_plot = y_test.detach().numpy()
    metabolites_corr = np.zeros(x_plot.shape[1])
    for i in range(x_plot.shape[1]):
        metabolites_corr[i] = spearmanr(x_plot[:,i], y_plot[:,i])[0]
    
    return test_error[-1].item(), np.nanmean(metabolites_corr)

def cross_validation(X_train, y_train, Nlayers_list, layer_size_list, weight_decay_list, dropout_list):
    from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
    from sklearn.model_selection import KFold
    import time
    start = time.time()
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X_train)


    test_error_for_all_hyperparameters = np.zeros((len(Nlayers_list), len(layer_size_list), 
                                                   len(weight_decay_list), len(dropout_list)))
    Spearmac_CC_for_all_hyperparameters = np.zeros((len(Nlayers_list), len(layer_size_list), 
                                                   len(weight_decay_list), len(dropout_list)))
    for i, Nlayers in enumerate(Nlayers_list):
        for j, layer_size in enumerate(layer_size_list):
            for k, weight_decay in enumerate(weight_decay_list):
                for l, dropout in enumerate(dropout_list):
                    #print("="*50)
                    #print(Nlayers, layer_size, weight_decay, dropout)
                    final_test_error_list = []
                    final_Spearmac_CC_list = []
                    for train_index, test_index in kf.split(X_train):
                        X_train_5fold, X_test_5fold = X_train[train_index], X_train[test_index]
                        y_train_5fold, y_test_5fold = y_train[train_index], y_train[test_index]
                        final_test_error, final_Spearmac_CC = MLP(Nlayers, layer_size, weight_decay, dropout, X_train_5fold, X_test_5fold, y_train_5fold, y_test_5fold)
                        final_test_error_list = final_test_error_list + [final_test_error]
                        final_Spearmac_CC_list = final_Spearmac_CC_list + [final_Spearmac_CC]
                    test_error_for_all_hyperparameters[i,j,k,l] = np.mean(final_test_error_list)
                    Spearmac_CC_for_all_hyperparameters[i,j,k,l] = np.mean(final_Spearmac_CC_list)
    stop = time.time()
    print("The computing time (s): {:.3f}".format(stop - start))
    return test_error_for_all_hyperparameters, Spearmac_CC_for_all_hyperparameters

def prediction(X_train, X_test, y_train, y_test, Nlayers, layer_size, weight_decay, dropout):
    #### convert to tensor
    X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
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
    y_pred = model(X_test)
    before_train = spearman(y_pred.squeeze(), y_test) 
    model.train()
    epoch = 1000
    test_error = []
    for epoch in range(epoch):
        # Loop over batches in an epoch using DataLoader
        for id_batch, (X_batch, y_batch) in enumerate(dataloader):
            #print(id_batch)
            # Forward pass
            y_batch_pred = model(X_batch)
            # Compute Loss
            loss = criterion(y_batch_pred.squeeze(), y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%20==0:
            model.eval()
            y_pred = model(X_test)
            #after_train = criterion(y_pred.squeeze(), y_test) 
            after_train = spearman(y_pred.squeeze(), y_test) 
            test_error = test_error + [after_train]
            if (loss < 0.1) and (test_error[-1] - test_error[-2]) < 0:
                model = copy.deepcopy(previous_model)
                break
            previous_model = copy.deepcopy(model)

    #### Evaluate the model
    model.eval()
    y_pred = model(X_test)
    after_train = spearman(y_pred.squeeze(), y_test) 
    return model, y_pred



# ### 5-fold cross-validation of predicting microbiome compostion at time t+1
if withBaselineMetabolome==True:
    X_train = X_train_ori.copy()
    X_test = X_test_ori.copy()
    y_train = y_train_ori[:, :-Nm]
    y_test = y_test_ori[:, :-Nm]
else:
    X_train = X_train_ori.copy()
    X_test = X_test_ori.copy()
    X_train = np.concatenate([X_train[:, :-Nm-Ni], X_train[:, -Ni:]], axis=1)
    X_test = np.concatenate([X_test[:, :-Nm-Ni], X_test[:, -Ni:]], axis=1)
    y_train = y_train_ori[:, :-Nm]
    y_test = y_test_ori[:, :-Nm]

Nlayers = 6
layer_size = 2048
weight_decay = 0
dropout = 0.0


from sklearn.preprocessing import StandardScaler 
scaler_x_1 = StandardScaler()  
scaler_x_1.fit(X_train)  
X_train = scaler_x_1.transform(X_train)  
X_test = scaler_x_1.transform(X_test)  
scaler_y_1 = StandardScaler()  
scaler_y_1.fit(y_train)
y_train = scaler_y_1.transform(y_train)  
y_test = scaler_y_1.transform(y_test)  

#### convert to tensor
X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test); y_test = torch.FloatTensor(y_test)

model, y_pred = prediction(X_train, X_test, y_train, y_test, Nlayers, layer_size, weight_decay, dropout)

import copy
model_1 = copy.deepcopy(model)
X_train_1 = copy.deepcopy(X_train)
X_test_1 = copy.deepcopy(X_test)
y_train_1 = copy.deepcopy(y_train)
y_test_1 = copy.deepcopy(y_test)


# ### 5-fold cross-validation of predicting metabolome at time t+1 based on microbiome compostion at time t+1
if withBaselineMetabolome==True:
    X_train = np.concatenate([scaler_y_1.inverse_transform(model_1(X_train_1).detach().numpy()), X_train_ori[:, -Nm-Ni:]], axis=1)
    X_test = np.concatenate([scaler_y_1.inverse_transform(model_1(X_test_1).detach().numpy()), X_test_ori[:, -Nm-Ni:]], axis=1)
    y_train = y_train_ori[:, -Nm:]
    y_test = y_test_ori[:, -Nm:]
else:
    X_train = np.concatenate([scaler_y_1.inverse_transform(model_1(X_train_1).detach().numpy()), X_train_ori[:, -Ni:]], axis=1)
    X_test = np.concatenate([scaler_y_1.inverse_transform(model_1(X_test_1).detach().numpy()), X_test_ori[:, -Ni:]], axis=1)
    y_train = y_train_ori[:, -Nm:]
    y_test = y_test_ori[:, -Nm:]

Nlayers = 6
layer_size = 2048
weight_decay = 0
dropout = 0.0

from sklearn.preprocessing import StandardScaler 
scaler_x_2 = StandardScaler()  
scaler_x_2.fit(X_train)  
X_train = scaler_x_2.transform(X_train)  
X_test = scaler_x_2.transform(X_test)  
scaler_y_2 = StandardScaler()  
scaler_y_2.fit(y_train)
y_train = scaler_y_2.transform(y_train)  
y_test = scaler_y_2.transform(y_test)  

#### convert to tensor
X_train = torch.FloatTensor(X_train); y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test); y_test = torch.FloatTensor(y_test)

model, y_pred = prediction(X_train, X_test, y_train, y_test, Nlayers, layer_size, weight_decay, dropout)

import copy
model_2 = copy.deepcopy(model)
X_train_2 = copy.deepcopy(X_train)
X_test_2 = copy.deepcopy(X_test)
y_train_2 = copy.deepcopy(y_train)
y_test_2 = copy.deepcopy(y_test)

from scipy.stats import spearmanr
x_plot = y_pred.detach().numpy()
y_plot = y_test.detach().numpy()
metabolites_corr = np.zeros(Nm)
for i in range(Nm):
    metabolites_corr[i] = spearmanr(x_plot[:,i+x_plot.shape[1]-Nm], y_plot[:,i+x_plot.shape[1]-Nm])[0]

metabolites_corr_annotated = metabolites_corr[compound_name==compound_name]
compound_name_annotated = compound_name[compound_name==compound_name]

if Ni==1:
    i_diet_perturbed = -1

    #### Evaluate the model
    model_1.eval()
    y_pred = model_1(X_train_1)

    X_train_perturbed = X_train_1.clone()
    X_train_perturbed[:, i_diet_perturbed] = 0
    #### Evaluate the model
    model_1.eval()
    y_pred_perturbed = model_1(X_train_perturbed)
    sensitivity_consumption = (y_pred_perturbed - y_pred).detach().numpy() / (X_train_perturbed[:, i_diet_perturbed] - X_train_1[:, i_diet_perturbed]).detach().numpy()[:,np.newaxis]
else:
    microbe_name = np.array(range(X_train_2.shape[1]-Ni-Nm))
    sensitivity_consumption = np.zeros((microbe_name.shape[0], Ni))
    for i_diet_perturbed in range(Ni):
        i_diet_perturbed = -Ni+i_diet_perturbed

        #### Evaluate the model
        model_1.eval()
        y_pred = model_1(X_train_1)

        X_train_perturbed = X_train_1.clone()
        X_train_perturbed[:, i_diet_perturbed] = 0
        #### Evaluate the model
        model_1.eval()
        y_pred_perturbed = model_1(X_train_perturbed)
        sensitivity_consumption_prelim = (y_pred_perturbed - y_pred).detach().numpy() / (X_train_perturbed[:, i_diet_perturbed] - X_train_1[:, i_diet_perturbed]).detach().numpy()[:,np.newaxis]
        sensitivity_consumption[:,i_diet_perturbed] = np.mean(sensitivity_consumption_prelim, 0)


compound_name = np.array(range(y_train_2.shape[1]))
microbe_name = np.array(range(X_train_2.shape[1]-Ni-Nm))
sensitivity_all = np.zeros((microbe_name.shape[0], compound_name.shape[0]))
for i_species_perturbed in range(microbe_name.shape[0]):
    #### Evaluate the model
    model_2.eval()
    y_pred = model_2(X_train_2)

    X_train_perturbed = X_train_2.clone()
    X_train_perturbed[:, i_species_perturbed] = 0
    #### Evaluate the model
    model_2.eval()
    y_pred_perturbed = model_2(X_train_perturbed)

    sensitivity = (y_pred_perturbed - y_pred).detach().numpy()  / (X_train_perturbed[:, i_species_perturbed] - X_train_2[:, i_species_perturbed]).detach().numpy()[:,np.newaxis]
    sensitivity_all[i_species_perturbed, :] = np.mean(sensitivity, 0)

########### Save data
pd.DataFrame(data=sensitivity_consumption.mean(0).reshape(-1,1), index=df_y_test_ori.columns[:-Nm], columns=df_X_test_ori.columns[-Ni:]).to_csv("./results/sensitivity_diet_and_microbes.csv")
pd.DataFrame(data=sensitivity_all, index=df_y_test_ori.columns[:-Nm], columns=df_y_test_ori.columns[-Nm:]).to_csv("./results/sensitivity_microbes_and_metabolites.csv")

print("The sensitivities are saved.")
