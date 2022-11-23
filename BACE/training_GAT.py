import random
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
#from models.gat import GATNet
from models.gcn import GCNNet
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn import metrics
import pandas as pd
import numpy as np
import os
#import paddle.nn as nn

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def train(model, device, loader_train, optimizer, epoch):
#    print('Training on {} samples...'.format(len(loader_train[0].dataset)))
    model.train()
    
    arr_data = []
    for batch_idx, data in enumerate(loader_train):
        arr_data_tmp = []
        for idx, data_data in enumerate(data):
#            print(data_data[0].y)
            arr_data_tmp.append(data_data)
#        print(arr_data_tmp[1])
        arr_data.append(arr_data_tmp)
#    print(len(arr_data[0]))
    for i in range(len(arr_data[0])):
#        print(len(arr_data[0]))
#        print(arr_data[i][0].y)
#        y = arr_data[0][i].y.view(-1, 1).long().to(device)
#        y = arr_data[i][0].y
#        y = y.squeeze(1)
        optimizer.zero_grad()
        
        arr_data_data = [a[i] for a in arr_data]
#        print()
#        print(arr_data_data)
#        print(i)
#        y = arr_data_data[0].y.view(-1, 1).long().to(device)
        y = arr_data_data[0].y.to(device)
#        print(y)
#        print(arr_data_data[0].y.shape)
#        print(arr_data_data[0].x.shape)
#        print(device)
        output = model(arr_data_data,device)
#        print(output.shape)
#        output = output.to(torch.float64)
#        y = y.to(torch.float32)
#        y = y.squeeze(1)
#        output = output.squeeze(1)
#        print(output)
#        print(y)
#        print(output.is_cuda)
        loss = loss_fn(output, y)
        loss = torch.sum(loss)
#        print(loss)
#        print('loss', torch.ones_like(torch.sum(loss)))
        loss.backward()
        optimizer.step()
#        if i % 10 == 0:
#            print('Train epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * i / len(arr_data[0]),  loss.item()))

def predicting(model, device, loader_test):
#    print(loader_test)
    model.eval()
#    print('Make prediction for {} samples...'.format(len(loader_test[0].dataset)))
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    with torch.no_grad():
        arr_data = []
        for batch_idx, data in enumerate(loader_test):
            arr_data_tmp = []
            for idx, data_data in enumerate(data):
#                print(data_data.y)
                arr_data_tmp.append(data_data)
            arr_data.append(arr_data_tmp)
            
        for i in range(len(arr_data[0])): 
#            print(i)
            arr_data_data = [a[i] for a in arr_data] #这个是求arr_data数据的每一列
            ys = arr_data_data[0].y
            output = model(arr_data_data, device)
#            print("预测值：")
#            print(output)
#            print("标签值：")
#            print(arr_data[0])
#            print(arr_data[0][i].y)
#            print(arr_data[0][i].y.cpu())
#            ys = F.softmax(output, 1).to('cpu').data.numpy()
            ys = output.cpu().numpy()
#            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x, ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
#            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, arr_data[0][i].y.cpu()), 0)
#    print(total_labels.numpy().flatten())
#    print(total_preds.numpy().flatten())    
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

modeling = GCNNet

# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

data_train = TestbedDataset(root='data', dataset='bace_train')
data_valid = TestbedDataset(root='data', dataset='bace_validation')
data_test = TestbedDataset(root='data', dataset='bace_test')

#print(data_train[1])

TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 2501
NUM_NET = 5

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

length_train = int(len(data_train)/NUM_NET)
length_valid = int(len(data_valid)/NUM_NET)
length_test = int(len(data_test)/NUM_NET)


print("length of training set:",length_train)
print("length of validation set:",length_valid)
print("length of testing set:",length_test)

train_num = np.linspace(0, length_train-1, length_train)
valid_num = np.linspace(0, length_valid-1, length_valid)
test_num = np.linspace(0, length_test-1, length_test)


loader_train = []
loader_valid = []
loader_test = []
for a in range(NUM_NET):
    train_num_tmp = [int(n * NUM_NET + a) for n in train_num]
    valid_num_tmp = [int(n * NUM_NET + a) for n in valid_num]
    test_num_tmp = [int(n * NUM_NET + a) for n in test_num]
    
    data_train_tmp = data_train[train_num_tmp]
    data_valid_tmp = data_valid[valid_num_tmp]
    data_test_tmp = data_test[test_num_tmp]
    
    loader_train.append(DataLoader(data_train_tmp, batch_size=TRAIN_BATCH_SIZE, shuffle=None,drop_last=True))
    loader_valid.append(DataLoader(data_valid_tmp, batch_size=VALID_BATCH_SIZE, shuffle=None,drop_last=True))
    loader_test.append(DataLoader(data_test_tmp, batch_size=TEST_BATCH_SIZE, shuffle=None,drop_last=True))

for i in range(5):
    model = modeling().to(device)
    #    loss_fn = nn.MSELoss()
    #    loss_fn = nn.L1Loss()
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    #    best_r2 = 0
    best_auc = 0
    #    best_MAE = 1000
    for epoch in range(NUM_EPOCHS):
        train(model, device, loader_train, optimizer, epoch + 1)
    #    T, S, Y = predicting(model, device, loader_valid)
        T, S, Y = predicting(model, device, loader_test)
#        print(T)
#        print(S)
    #        MAE = mean_absolute_error(T,S)
    #        r2 = r2_score(T, S)
        AUC = roc_auc_score(T, S)
        if best_auc < AUC:
            best_auc = AUC
        if epoch % 20 == 0:
            print("-------------------------------------------------------")
            print("epoch:",epoch)
    #        print(AUC)
            print('best_auc', best_auc)
