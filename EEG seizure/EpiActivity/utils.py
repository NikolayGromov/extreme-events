from dictionary import *

import pyedflib
import mne
import numpy as np
import pandas as pd

import torchvision
import torchaudio
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.ticker as ticker
from os import listdir
import os

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


### READ SIGNAL ###

def ReadSignal(file_name): 

    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((20, f.getNSamples()[0])) #or n
    print(file_name, n)
    
    if n == 22:
        for i in np.arange(19):
            sigbufs[i, :] = f.readSignal(i)
        sigbufs[19, :] = f.readSignal(21)
    elif n == 23:
        for i in np.arange(19):
            sigbufs[i, :] = f.readSignal(i)
        sigbufs[19, :] = f.readSignal(20)
    elif n == 21:
        for i in np.arange(20):
            sigbufs[i, :] = f.readSignal(i)
    else:
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)

    time = [1/f.samplefrequency(0) * i for i in range(len(sigbufs[0]))]

    annotations = f.readAnnotations()  

    new_annot = [(annotations[0][i], annotations[1][i], annotations[2][i])  
                 for i in range(len(annotations[0])) 
                                if (annotations[1][i] > 0) and (annotations[2][i] in ["Ð´Ð°Ð±Ð» Ñ\x81Ð¿Ð°Ð¹Ðº", "*", "?", "F7", 
                                                                                      'Ð\xa03(blockUserBlock)', "F7(blockUserBlock)", 
                                                                                      "O2(blockUserBlock)", "F7(blockUserBlock)", 
                                                                                      'Ð¡4(blockUserBlock)', 'Ñ\x8dÐ°',
                                                                                      "F7(blockUserBlock)(blockUserBlock)"])]
    f.close()
    return sigbufs, new_annot, time, f.samplefrequency(0)


def NormalizeAndClip(data):
    for i in tqdm(range(len(data))):
        signal = data[i]
        means = signal.mean(axis=1)[..., None]
        stds = signal.std(axis=1)[..., None]
        signal = np.clip((signal - means) / stds, a_min=-10, a_max=10)
        data[i] = signal

### TRANSFORM TO ONE FREQUENCY ###

def Transform(sample_rate, records, freqs, times): # only for 199
    transform = 0
    new_freq = 0
    if sample_rate == 199:
        transform500 = torchaudio.transforms.Resample(250625, 100000)
        transform1000 = torchaudio.transforms.Resample(50125, 10000)
        
        new_freq = 80000 / 401
    else:
        transform = torchaudio.transforms.Resample(100000, 250625) 
        new_freq = 500
    for i in range(len(records)):
        if int(freqs[i]) != sample_rate:
            if int(freqs[i]) == 500:
                new_sigbufs = []
                sigbufs = records[i]
                for sig in tqdm(sigbufs):
                    new_sigbufs.append(transform500(torch.FloatTensor(sig)))
                new_sigbufs = np.array(new_sigbufs)
                records[i] = new_sigbufs
                freqs[i] = new_freq
                times[i] = [1/new_freq * j for j in range(len(new_sigbufs[0]))]   
            elif int(freqs[i]) == 1000:
                new_sigbufs = []
                sigbufs = records[i]
                for sig in tqdm(sigbufs):
                    new_sigbufs.append(transform1000(torch.FloatTensor(sig)))
                new_sigbufs = np.array(new_sigbufs)
                records[i] = new_sigbufs
                freqs[i] = new_freq
                times[i] = [1/new_freq * j for j in range(len(new_sigbufs[0]))]
    for i in range(len(freqs)):
        freqs[i] = 80000 / 401
        times[i] = [1/freqs[i] * j for j in range(len(records[i][0]))]

### LABELING ###

def Labeling(time, events_lst):
    labels = np.zeros_like(time)

    for events in events_lst:
        for event in tqdm(events):
            start = np.array(time < event[0]).argmin()
            fin = np.array(time < event[0] + event[1]).argmin()
            labels[start:fin] =  1
    return labels
        
### TRAIN TEST SPLIT ###

def Shuffle(records):
    shuffled_records = []
    for i in range(len(records)):
        shuffled_records.append(records[i].copy())
        np.random.shuffle(shuffled_records[-1][:-1])
    return shuffled_records

def GetTrainTestByIdxs(array, train_indices):
    train_array = [array[idx] for idx, is_train in enumerate(train_indices) if is_train == True]
    test_array = [array[idx] for idx, is_train in enumerate(train_indices) if is_train == False]    
    return train_array, test_array

def WrapTrainTestInput(train_records, train_annots, train_times, sneos_data_train, train_labels, 
                       test_records, test_annots, test_times, sneos_data_test, test_labels, shuffle_leads=False):
    
    sneos, train_sneos, train_mcs = sneos_data_train
    _, test_sneos, test_mcs = sneos_data_test

    if shuffle_leads:
        train_records = Shuffle(train_records)
    
    train_data = []
    for i in range(len(train_records)):
        train_time_start = train_annots[i][0][0]
        train_time_end = train_annots[i][-1][0]
        train_idx_start = (np.array(train_times[i]) < train_time_start).argmin()
        train_idx_fin = (np.array(train_times[i]) < train_time_end).argmin()
        
        if sneos is not None:
            train_data.append((torch.FloatTensor(train_records[i][:, train_idx_start:train_idx_fin]), 
                              torch.FloatTensor(train_sneos[i][:, train_idx_start:train_idx_fin]), 
                              torch.FloatTensor(train_mcs[i][:, train_idx_start:train_idx_fin])))
        else:
            train_data.append((torch.FloatTensor(train_records[i][:, train_idx_start:train_idx_fin]), None, None))
        
        current_labels = train_labels[i][train_idx_start:train_idx_fin]
        new_trainl = torch.zeros(2, len(current_labels))
        new_trainl = (torch.arange(2) == torch.LongTensor(current_labels)[:,None]).T
        new_trainl = new_trainl.float()
        train_labels[i] = new_trainl
    
    test_data = []
    for i in range(len(test_records)):
        test_time_start = test_annots[i][0][0]
        test_time_end = test_annots[i][-1][0]
        test_idx_start = (np.array(test_times[i]) < test_time_start).argmin()
        test_idx_fin = (np.array(test_times[i])< test_time_end).argmin()

        if sneos is not None:
            test_data.append((torch.FloatTensor(test_records[i][:, test_idx_start:test_idx_fin]), 
                              torch.FloatTensor(test_sneos[i][:, test_idx_start:test_idx_fin]), 
                              torch.FloatTensor(test_mcs[i][:, test_idx_start:test_idx_fin])))
        else:
            test_data.append((torch.FloatTensor(test_records[i][:, test_idx_start:test_idx_fin]), None, None))                  
        
        current_labels = test_labels[i][test_idx_start:test_idx_fin]
        new_testl = torch.zeros(2, len(current_labels))
        new_testl = (torch.arange(2) == torch.LongTensor(current_labels)[:,None]).T
        new_testl = new_testl.float()
        test_labels[i] = new_testl
    
    return train_data, train_labels, test_data, test_labels 

def GetTrainTestSplit(record_names, records, annots, times, labels, index, shuffle_leads, N, sneos=None, mcs=None):
    train_indices = np.ones(N) #number of records
    if N %2 == 1 and index == N // 2 + 1:
        train_indices[-1] = 0
    else:
        train_indices[2*index: 2*index+2] = 0
        
    train_sneos, test_sneos = None, None
    train_mcs, test_mcs = None, None
    
    train_records, test_records = GetTrainTestByIdxs(records, train_indices)
    if sneos is not None:
        train_sneos, test_sneos = GetTrainTestByIdxs(sneos, train_indices)
        train_mcs, test_mcs = GetTrainTestByIdxs(mcs, train_indices)
        
        
    train_labels, test_labels = GetTrainTestByIdxs(labels, train_indices)  
    train_annots, test_annots = GetTrainTestByIdxs(annots, train_indices)
    train_times, test_times = GetTrainTestByIdxs(times, train_indices)
    train_record_names, test_record_names = GetTrainTestByIdxs(record_names, train_indices)
    return WrapTrainTestInput(train_records, train_annots, train_times, [sneos, train_sneos, train_mcs], train_labels, 
                              test_records, test_annots, test_times, [sneos, test_sneos, test_mcs], test_labels, 
                              shuffle_leads) + [train_record_names, test_record_names]

#Creating samples
def LeadNamesToVector(record_leads, bipolar_names=LEAD_NAMES):
    result_vector = np.zeros(len(bipolar_names))
    for lead in record_leads["leading"]:
        for i, bipolar_lead in enumerate(bipolar_names):
            if lead in bipolar_lead:
                result_vector[i] = 1
    for lead in record_leads["additional"]:
        for i, bipolar_lead in enumerate(bipolar_names):
            if lead in bipolar_lead:
                result_vector[i] = 1
    return result_vector

def FindStartFinSpike(labels: torch.LongTensor, fin):
    start_local = ((labels[1] == 1).nonzero(as_tuple=True)[0])[0].item()
    #print(start_local)#[0].item()
    if labels[1, start_local:].min() == 1:
        fin_local = labels.shape[1]
    else:
        fin_local = ((labels[1][start_local:] == 0).nonzero(as_tuple=True)[0])[0].item()
    return start_local + fin, start_local + fin_local + fin

def CreateInoutSeqLocalization(x, labels, record_vector, rf, ov=0): # last in labels[1] should be 0 ???
    inout_seq = []
    fin=0
    train_seq = train_label = None
    while fin < labels.shape[1] and labels[1, fin:].max() == 1:
        start, fin = FindStartFinSpike(labels[:, fin:], fin)
        padding = rf - fin + start
        if start - padding//2 - 1 < 0:
            train_seq = x[:, start: fin + padding]
            train_label = labels[1, start: fin + padding]
        elif fin + padding//2 >= labels.shape[1]:
            train_seq = x[:, start - padding: fin]
            train_label = labels[1, start - padding: fin]
        else:
            train_seq = x[:, start - padding//2 - 1: fin + padding//2][:, :rf]
            train_label = labels[1, start - padding//2 - 1: fin + padding//2][:rf]
        assert train_seq.shape[1] == train_label.shape[0] == rf
        assert train_label.max() == 1
        inout_seq.append((train_seq, train_label.unsqueeze(0), "None", record_vector)) # record vector is not TorchTensor (Is a problem?)
    return inout_seq

RECEPTIVE_FIELD = 4000
OVERLAP = 0 

def CreateSamples(x, labels, rf = RECEPTIVE_FIELD, ov = OVERLAP, sneo=None, mc=None):
    inout_seq = []
    L = x.shape[-1]
    if sneo != None:
        for i in tqdm(range(ov, L- rf - ov, rf)):
            train_seq = x[:, i-ov:i+rf+ov]
            train_sneo = sneo[:, i-ov:i+rf+ov]
            train_mc = mc[:, i-ov:i+rf+ov]
            
            train_label = labels[:, i:i+rf]
            inout_seq.append((train_seq, train_sneo, train_mc, train_label))
    else:
        for i in tqdm(range(ov, L- rf - ov, rf)):
            train_seq = x[:, i-ov:i+rf+ov]
            train_label = labels[:, i:i+rf]
            inout_seq.append((train_seq, "None", "None", train_label)) #with real None problems with dataloader
  
    return inout_seq
    

    ### MODELS AND TRAINING ###
class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, 
                               padding = int(np.ceil(dilation * (kernel_size-1) / 2)), bias=True) # for stride=1, else need to calculate and change
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        inp_shape = int(np.ceil(x.shape[2] / self.stride))
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)[:, :, :inp_shape] 
        #print("conbr_out", out.shape)
        return out      

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)        
        x_out = torch.add(x, x_re)
        return x_out          

class UNET_1D(nn.Module):
    def __init__(self ,input_dim,layer_n,kernel_size, n_down_layers, depth, n_features=1): # n_features for additional features in some other exps
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.n_down_layers = n_down_layers
        self.depth = depth
        
        self.AvgPool1D = nn.ModuleList([nn.AvgPool1d(input_dim, stride=5**i, padding=8) for i in range(1, self.n_down_layers)])
        
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, depth)
        self.layer1_sneo = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, self.depth)
        self.layer1_mc = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, self.depth)
        
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, self.depth)
        
        self.down_layers = nn.ModuleList([self.down_layer(self.layer_n*(1+i)+n_features*self.input_dim, self.layer_n*(2+i), 
                                            self.kernel_size,5, self.depth) for i in range(1, self.n_down_layers)])


        self.cbr_up = nn.ModuleList([conbr_block(int(self.layer_n*(2*i+1)), int(self.layer_n*i), self.kernel_size, 1, 1) 
                       for i in range(self.n_down_layers, 0, -1)]) #input size is a sizes sum of outs of 2 down layers for current down depth
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest') 
        
        self.outcov = nn.Conv1d(self.layer_n, 2, kernel_size=self.kernel_size, stride=1,
                                padding = int(np.ceil(1 * (self.kernel_size-1) / 2)))
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
        
        
            
    def forward(self, x):
        inp_shape = x.shape[2]
        
        
        
        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        outs = [out_0, out_1]
        for i in range(self.n_down_layers-1):
            pool = self.AvgPool1D[i](x)
            x_down = torch.cat([outs[-1],pool],1)
            outs.append(self.down_layers[i](x_down))




        #############Decoder####################
        up = self.upsample(outs[-1])[:, :, :outs[-2].shape[2]]
        for i in range(self.n_down_layers):
                        
            up = torch.cat([up,outs[-2-i]],1)
            up = self.cbr_up[i](up)
            if i + 1 < self.n_down_layers:
                up = self.upsample(up)[:, :, :outs[-3-i].shape[2]]

        out = self.outcov(up)


        return out[:, :, :inp_shape] 
    
class UNET_1D_Localization(UNET_1D):
    def __init__(self, window_len, input_dim, layer_n, kernel_size, n_down_layers, depth, n_features=1): # n_features for additional features in some other exps
        super(UNET_1D_Localization, self).__init__(input_dim, layer_n, kernel_size, n_down_layers, depth, n_features)
        self.classifier1 = nn.Linear(window_len, 256)
        self.classifier2 = nn.Linear(256, 1)
        self.classifier_features = nn.Linear(self.layer_n + 1, self.input_dim)
        self.relu = nn.ReLU()


    def forward(self, x, labels):
        inp_shape = x.shape[2]
        
        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        outs = [out_0, out_1]
        for i in range(self.n_down_layers-1):
            pool = self.AvgPool1D[i](x)
            x_down = torch.cat([outs[-1],pool],1)
            outs.append(self.down_layers[i](x_down))




        #############Decoder####################
        up = self.upsample(outs[-1])[:, :, :outs[-2].shape[2]]
        for i in range(self.n_down_layers):
                        
            up = torch.cat([up,outs[-2-i]],1)
            up = self.cbr_up[i](up)
            if i + 1 < self.n_down_layers:
                up = self.upsample(up)[:, :, :outs[-3-i].shape[2]]

        #############Classification#############
        x = torch.cat([up, labels], 1)
        x = x.permute(0, 2, 1)
        x = self.classifier_features(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.classifier1(x)
        x = self.relu(x)
        x = self.classifier2(x)
        return x[:, :, 0]

        
class MDB_block(nn.Module):
    def __init__(self, k, F):
        super(MDB_block, self).__init__()

        #print(F, k)
        self.conv11 = nn.Conv1d(F, F//2, kernel_size=k, dilation=1, padding=(k-1)//2) #p = d * (k-1) / 2
        self.bn11 = nn.BatchNorm1d(F//2)
        self.conv12 = nn.Conv1d(F, F//4, kernel_size=k, dilation=2, padding=k-1) 
        self.bn12 = nn.BatchNorm1d(F//4)
        self.conv13 = nn.Conv1d(F, F//8, kernel_size=k, dilation=4, padding=2*(k-1))
        self.bn13 = nn.BatchNorm1d(F//8)
        self.conv14 = nn.Conv1d(F, F//8, kernel_size=k, dilation=8, padding=4*(k-1))
        self.bn14 = nn.BatchNorm1d(F//8)

        self.conv21 = nn.Conv1d(F//2, F//2, kernel_size=k, dilation=1, padding=(k-1)//2)
        self.bn21 = nn.BatchNorm1d(F//2)
        self.conv22 = nn.Conv1d(F//4, F//4, kernel_size=k, dilation=2, padding=k-1)
        self.bn22 = nn.BatchNorm1d(F//4)
        self.conv23 = nn.Conv1d(F//8, F//8, kernel_size=k, dilation=4, padding=2*(k-1))
        self.bn23 = nn.BatchNorm1d(F//8)
        self.conv24 = nn.Conv1d(F//8, F//8, kernel_size=k, dilation=8, padding=4*(k-1))
        self.bn24 = nn.BatchNorm1d(F//8)

        self.relu = nn.ReLU()

        
    
    def forward(self,x):

        x1 = self.conv11(x)
        x1 = self.bn11(x1)
        x1 = self.relu(x1)
        x1 = self.conv21(x1)
        x1 = self.bn21(x1)
        x1 = self.relu(x1)
        
        x2 = self.conv12(x)
        x2 = self.bn12(x2)
        x2 = self.relu(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu(x2)

        x3 = self.conv13(x)
        x3 = self.bn13(x3)
        x3 = self.relu(x3)
        x3 = self.conv23(x3)
        x3 = self.bn23(x3)
        x3 = self.relu(x3)

        x4 = self.conv14(x)
        x4 = self.bn14(x4)
        x4 = self.relu(x4)
        x4 = self.conv24(x4)
        x4 = self.bn24(x4)
        x4 = self.relu(x4)
        #print("x1", x1.size())
        #print("x2", x2.size())
        #print("x3", x3.size())
        #print("x4", x4.size())
        
        return torch.concatenate([x1, x2, x3, x4], axis=1)  

class SEED(nn.Module):
    def __init__(self, overlap=0, input_dim=27, F = 64, q1=0.2, q2=0.5, N1=256, N2=128):
        super(SEED, self).__init__()

        self.N1 = N1
        
        self.bn = nn.BatchNorm1d(input_dim)
        self.conv1 = nn.Conv1d(input_dim, F, kernel_size=3, padding = 520 - overlap)
        self.bn1 = nn.BatchNorm1d(F)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(F, 2*F, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(2*F)

        self.AvgPool1D = nn.AvgPool1d(2)

        self.mdb1 = MDB_block(3, 2*F)

        self.conv3 = nn.Conv1d(2*F, 4*F, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(4*F)
        
        self.mdb2 = MDB_block(3, 4*F)

        self.dropout1 = nn.Dropout(q1)
        self.dropout2 = nn.Dropout(q2)

        
        self.lstm1 = nn.LSTM(N1, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.convlstm1 = nn.Conv1d(2*N1, N1, kernel_size=1) #256?
        self.bn3 = nn.BatchNorm1d(N1)

        self.lstm2 = nn.LSTM(N1, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.convlstm2 = nn.Conv1d(2*N1, N1, kernel_size=1) #256?
        self.bn4 = nn.BatchNorm1d(N1)

        self.classifier1 = nn.Conv1d(N1, N2, kernel_size=1) 
        self.bn5 = nn.BatchNorm1d(N2)
        self.classifier2 = nn.Conv1d(N2, 2, kernel_size=1) 

        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')

    def forward(self,x):
        x = self.bn(x)
        x = self.conv1(x)
        #print("conv1", x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        #print("conv2", x.size())
        
        x = self.bn2(x)
        x = self.relu(x)

        x = self.AvgPool1D(x)
        #print("AvgPool1D1", x.size())
        
        x = self.mdb1(x)
        #print("mdb1", x.size())

        x = self.conv3(x)
        #print("conv3", x.size())
        
        x = self.bn3(x)
        x = self.relu(x)

        x = self.AvgPool1D(x)
        #print("AvgPool1D2", x.size())
        
        x = self.mdb2(x)
        
        #print("mdb2", x.size())

        x = self.AvgPool1D(x)
        #print("AvgPool1D3", x.size())

        x = x[:, :, 65:-64]
        #print("x crop", x.size())

        x = self.dropout1(x)

        lstm_out, (h_n, c_n) = self.lstm1(x.transpose(2, 1))

        #print("lstm_out1", lstm_out.size())

        x = lstm_out[:, :, :self.N1] + lstm_out[:, :, self.N1:]
        x = self.dropout2(x)

        #print("x.size()", x.size())

        lstm_out, (h_n, c_n) = self.lstm2(x)

        #print("lstm_out2", lstm_out.size())

        x = lstm_out[:, :, :self.N1] + lstm_out[:, :, self.N1:]
        x = self.dropout2(x)
        #print("x.size()", x.size())
        x = x.transpose(2, 1)
        x = self.dropout2(x)
        #print("x.size()", x.size())

        x = self.classifier1(x)
        #print("class1", x.size())
        
        x = self.bn5(x)
        x = self.relu(x)

        x = self.classifier2(x)    
        #print("class2", x.size())

        x =self.upsample(x)    
        #print("upsample", x.size())
    
        return x
        
        
def run_epoch(model, optimizer, criterion, dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, sneo_or_labels, mc, y) in tqdm(enumerate(dataloader)):
        if is_training:
            optimizer.zero_grad()
        out = None
        if sneo_or_labels[0] != "None" and mc[0] != "None":
            
            out = model(x.to('cuda'), sneo_or_labels.to("cuda"), mc.to("cuda"))
        elif sneo_or_labels[0] != "None":
            #x, y = sample
            out = model(x.to("cuda"), sneo_or_labels.to("cuda"))
        else:
            out = model(x.to("cuda"))
        loss = criterion(out, y.to('cuda'))

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / len(dataloader))


    return epoch_loss

def GetRawMetricsAndCMEPINoLogging(predictions, test_labels):
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0    
    
    for i in range(len(test_labels)):
        pred_len = len(predictions[i])

        TP, FP, FN = CollectingTPFPFN(predictions[i], test_labels[i][1, :pred_len].numpy())

        TP_sum += TP
        FP_sum += FP
        FN_sum += FN      

        
    return 2 * TP_sum / (2 * TP_sum + FP_sum + FN_sum) if TP_sum > 0 else 0


def CalculateMetric(model, test_dataloader):
    

    all_preds = []
    record_preds = []
    answers = []
    all_answers = []
    
    for idx, (x, sneo, mc, y) in tqdm(enumerate(test_dataloader)):
        
        out = None
        if sneo[0] != "None":
            out = model(x.to('cuda'), sneo.to("cuda"), mc.to("cuda"))
        else:
            out = model(x.to("cuda"))
        
        m = nn.Softmax(dim=1)
        out = m(out)
            
        preds = np.argmax(out.detach().cpu().numpy(), axis=1)
        record_preds.append(preds)
        answers.append(y.detach().cpu().numpy()[:, 1])
    shapes = np.array(record_preds).shape
    
    record_preds = np.array(record_preds).reshape(shapes[0] * shapes[1] * shapes[2])
    answers = torch.LongTensor(np.vstack([np.zeros(shapes[0] * shapes[1] * shapes[2]), 
                         np.array(answers).reshape(shapes[0] * shapes[1] * shapes[2])]))
    
    all_preds.append(record_preds)
    all_answers.append(answers)
    
    
    threshold1 = 20
    threshold2 = None

    
    for j in range(len(all_preds)):
        PostProcessing(all_preds[j], threshold1, threshold2)
    metric = GetRawMetricsAndCMEPINoLogging(all_preds, all_answers)
    return metric

def Train(model, train_dataloader, test_dataloader, i, path, localization=False):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, patience = 3, factor=0.5, min_lr=0.00001)
    epochs = 250

    losses_train = []
    losses_test = []
    metrics = []
    best_loss = 10e9
    best_metric = 0
    if localization:
        best_metric = -best_loss
    
    # begin training

    early_stop_count = 40
    current_es = 0
    best_epoch = 0
    for epoch in range(epochs): 


        loss_train = run_epoch(model, optimizer, criterion, train_dataloader, is_training=True)
        loss_val = run_epoch(model, optimizer, criterion, test_dataloader)
        scheduler.step(loss_val)
        losses_train.append(loss_train)
        losses_test.append(loss_val)
        
        if not localization:
            metric = CalculateMetric(model, test_dataloader)

            metrics.append(metric)
        else:
            metrics.append(-loss_val)

        
        if metrics[-1] >= best_metric:
            best_metric =  metrics[-1]
            torch.save(model.state_dict(), path + "/Split" + str(i) +"/Unet1d")
            best_epoch = epoch
            current_es = 0
        else:
            current_es += 1
        
        if current_es >= early_stop_count:
            break


        clear_output(True)
        fig = plt.figure(figsize=(10, 9))

        ax_1 = fig.add_subplot(3, 1, 1)
        ax_2 = fig.add_subplot(3, 1, 2)
        ax_3 = fig.add_subplot(3, 1, 3)
        
        ax_1.set_title('train loss')
        ax_1.plot(losses_train)
        ax_2.set_title('test loss')
        ax_2.plot(losses_test)
        ax_3.set_title('test metric')
        ax_3.plot(metrics)
        plt.savefig(path + "/Split" + str(i) + "/Unet1dFigure")

        plt.show()



        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f}'
                  .format(epoch+1, epochs, loss_train, loss_val))
    f = open(path + "/Split" + str(i) + "/BestEpoch.txt", 'w')
    f.write(str(best_epoch))
    f.close()


### COLLECTING PREDICTIONS, CALCULATING METRIC ###

def CollectingPreds(model, test_data, return_logits=False):

    model.eval()
    model.cpu()
    all_preds = []
    for i in range(len(test_data)):
        record_preds = []
        for idx in tqdm(range(OVERLAP, test_data[i][0].size()[1]- RECEPTIVE_FIELD - OVERLAP, RECEPTIVE_FIELD)):

            test_seq = test_data[i][0][:, idx-OVERLAP:idx+RECEPTIVE_FIELD+OVERLAP][None, ...]
            out = None
            if test_data[i][1] is not None:
                test_sneo_seq = test_data[i][1][:, idx-OVERLAP:idx+RECEPTIVE_FIELD+OVERLAP][None, ...]
                test_mc_seq = test_data[i][2][:, idx-OVERLAP:idx+RECEPTIVE_FIELD+OVERLAP][None, ...] 
                out = model(test_seq, test_sneo_seq, test_mc_seq)
            else:
                out = model(test_seq)
                   
            m = nn.Softmax(dim=1)
            out = m(out)
            
            if return_logits:
                preds = out.detach().cpu().numpy()[:, 1]
                record_preds.append(preds)
            else:
                preds = np.argmax(out.detach().cpu().numpy(), axis=1)
                record_preds.append(preds)
        shapes = np.array(record_preds).shape
        record_preds = np.array(record_preds).reshape(shapes[0] * shapes[1] * shapes[2])

        all_preds.append(record_preds)
    return all_preds

def CollectingPredsLocalization(model, test_data, test_labels, moscow_leads, test_record_names, window_len=500):
    model.eval()
    #model.cpu()
    all_preds = []
    all_answers = []
    for i in range(len(test_data)):
        record_preds = []
        record_answers = []
        test_seq = CreateInoutSeqLocalization(test_data[i][0], test_labels[i], 
                                              LeadNamesToVector(moscow_leads[test_record_names[i].split("/")[-1]]), window_len)
        test_dataloader = DataLoader(test_seq, batch_size=32, shuffle=False, drop_last=True)
        for (x, label, _, y) in test_dataloader:
            out = model(x.to("cuda"), label.to("cuda")) 
                          

            out = out >= 0

            record_preds.append(out.detach().cpu().numpy())
            record_answers.append(y.detach().cpu().numpy())
        shapes = np.array(record_preds).shape
        print(shapes)
        record_preds = np.array(record_preds).reshape(shapes[0] * shapes[1], shapes[2])
        record_answers = np.array(record_answers).reshape(shapes[0] * shapes[1], shapes[2])
        all_preds.append(record_preds)
        all_answers.append(record_answers)

    return all_preds, all_answers

def MergeClose(predictions, threshold):
    i = 0
    in_event = False
    while i < len(predictions):
        while i < len(predictions) and predictions[i] == 1:
            in_event = True
            i += 1
        if  i < len(predictions) and in_event:
            if np.any(predictions[i:i+threshold]):
                while  i < len(predictions) and predictions[i] == 0:
                    predictions[i] = 1
                    i += 1
            else:
                in_event = False
        i += 1

def DeleteShortEvents(predictions, threshold):
    i = 0
    while i < len(predictions):
        event_len = 0
        event_idx_start = i
        while i < len(predictions) and predictions[i] == 1:
            i += 1
            event_len += 1
        if event_len < threshold:
            predictions[event_idx_start:i] = 0
        i += 1
def PostProcessing(predictions, threshold1, threshold2=None):
    MergeClose(predictions, threshold1)
    if threshold2 is None:
        DeleteShortEvents(predictions, threshold1)
    else:
        DeleteShortEvents(predictions, threshold2)

def CollectingTPFPFN(pred_labels, true_labels):
    i = 0
    TP = 0
    FP = 0
    FN = 0

    is_true_flag = 0
    is_pred_flag = 0
    is_used_pred_flag = 0
    
    while i < len(pred_labels):
        if pred_labels[i] == 0:
            is_used_pred_flag = 0
        while i < len(pred_labels) and true_labels[i] == 1:
            is_true_flag = 1
            if not is_used_pred_flag:
                if pred_labels[i] == 1:
                    is_pred_flag = 1
                    is_used_pred_flag = 1 
            else:
                if pred_labels[i] == 0:
                    is_used_pred_flag = 0
            i += 1
        if is_true_flag:
            if is_pred_flag:
                TP += 1
            else:
                FN += 1
            i -= 1

        
        is_true_flag = 0
        is_pred_flag = 0   
        i += 1

    i = 0
    while i < len(pred_labels):
        while i < len(pred_labels) and pred_labels[i] == 1:
            is_pred_flag = 1
            if true_labels[i] == 1:
                is_true_flag = 1
            i += 1
        if is_pred_flag and not is_true_flag:
            FP += 1
        is_pred_flag = 0
        is_true_flag = 0
        i += 1

    return TP, FP, FN 


def GetRawMetricsAndCMEPI(predictions, test_labels, split_index, path, N):
    lens = []
    sums = []

    acc = []
    precision = []
    recall = []
    f1 = []

    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    
    all_cm = 0
    
    train_indices = np.ones(N) #number of records
    if N %2 == 1 and split_index == N // 2 + 1:
        train_indices[-1] = 0
    else:
        train_indices[2*split_index: 2*split_index+2] = 0
        
    train_record_names, test_record_names = GetTrainTestByIdxs(record_names, train_indices)
    
        
    f = open(path + "/Split" + str(split_index) + "/Metrics.txt", 'w')
    
    for i in range(len(test_labels)):
        pred_len = len(predictions[i])

        TP, FP, FN = CollectingTPFPFN(predictions[i], test_labels[i][1, :pred_len].numpy())

        TP_sum += TP
        FP_sum += FP
        FN_sum += FN
            

        if TP + FP != 0:
            precision.append(TP / (TP + FP))
        else:
            precision.append(0)
        recall.append(TP / (TP + FN))
        f1.append(2 * TP / (2 * TP + FP + FN))
        

        cm = np.array([[0, FP], [FN, TP]])
        all_cm += cm
        
        f.write("=============Record " + test_record_names[i] + "================\n")
    
        f.write("precision " + str(precision[i]) + "\n")
        f.write("recall " + str(recall[i]) + "\n")
        f.write("f1 score " + str(f1[i]) + "\n")
                
        con_mat = ConfusionMatrixDisplay(cm)
        con_mat.plot().figure_.savefig(path + "/Split" + str(split_index) + 
                                "/ConfusionMatrix_" + test_record_names[i].replace("/", "_") + ".png")
    f.write("===========ALL RECORDS SCORE==================\n")
    
    
    if TP_sum + FP_sum != 0:
        f.write("Full precision " + str(TP_sum / (TP_sum + FP_sum)) + "\n")
    else:
        f.write("Full precision " + str(0) + "\n")
    f.write("Full recall " + str(TP_sum / (TP_sum + FN_sum)) + "\n")
    f.write("Full f1 " + str(2 * TP_sum / (2 * TP_sum + FP_sum + FN_sum)) + "\n")
    
    
    f.close()
    con_mat = ConfusionMatrixDisplay(all_cm)
    con_mat.plot().figure_.savefig(path + "/Split" + str(split_index) + 
                                "/ConfusionMatrixFull.png")

def CalculateSWI(labels, sampling_rate=500, area_SWI=False):
    if not area_SWI:
        event_num = 0
        silence_num = 0
        for i in range(0, len(labels), sampling_rate):
            if labels[i:i+sampling_rate].max() == 1:
                event_num += 1
            else:
                silence_num += 1
        return event_num / (event_num + silence_num)
    return labels.sum() / len(labels)

### LOGGING AND CREATING FOLDER ###

def LogResults(model, test_data, test_labels, i, path, last_epoch, low_freq, write_edf, annots, area_SWI, N):
    if not last_epoch and test_data[0][1] is not None:
        model.load_state_dict(torch.load(path + "/Split" + str(i) +"/Unet1d"))  
    elif not last_epoch:
        model.load_state_dict(torch.load(path + "/Split" + str(i) +"/Unet1d"), strict=False)
    all_preds = CollectingPreds(model, test_data)
    threshold = 20 #low because of lower sample rate, for 500 need to up to 30
    sr = 200
    if not low_freq:
        threshold = 30
        sr = 500
    for j in range(len(all_preds)):
        PostProcessing(all_preds[j], threshold)
    GetRawMetricsAndCMEPI(all_preds, test_labels, i, path, N)  
    
    if write_edf:
        WriteEDFCrossValid(all_preds, i, annots, path, N)
        
    SWIs_pred = []
    SWIs_true = []
    
    for j in range(len(all_preds)):
        SWIs_pred.append(CalculateSWI(all_preds[j], sr, area_SWI))
        SWIs_true.append(CalculateSWI(test_labels[j][1], sr, area_SWI))     
    np.savetxt(path + "/Split" + str(i) +"/SWIPred", np.array(SWIs_pred))
    np.savetxt(path + "/Split" + str(i) +"/SWITrue", np.array(SWIs_true))   

def LogResultsLocalization(model, test_data, test_labels, test_record_names, moscow_leads, split_index, path, window_len):
    
    model.load_state_dict(torch.load(path + "/Split" + str(split_index) +"/Unet1d"))  
 
    
    all_preds, all_answers = CollectingPredsLocalization(model, test_data, test_labels, moscow_leads, test_record_names, window_len)
    
    # metrics calculation
    precisions = []
    recalls = []
    f1s = []
    f = open(path + "/Split" + str(split_index) + "/Metrics.txt", 'w')

    for i in range(len(all_preds)):
        precisions.append(precision_score(all_answers[i], all_preds[i], average='macro'))
        recalls.append(recall_score(all_answers[i], all_preds[i], average='macro'))
        f1s.append(f1_score(all_answers[i], all_preds[i], average='macro'))
        f.write("=============Record " + test_record_names[i] + "================\n")
    
        f.write("precision " + str(precisions[i]) + "\n")
        f.write("recall " + str(recalls[i]) + "\n")
        f.write("f1 score " + str(f1s[i]) + "\n")
                
        
    f.write("===========ALL RECORDS SCORE==================\n")
    f.write("precision " + str(np.mean(precisions)) + "\n")
    f.write("recall " + str(np.mean(recalls)) + "\n")
    f.write("f1 score " + str(np.mean(f1s)) + "\n")
    f.close()

    # record leads localization
    for i in range(len(all_preds)):
        leads_dist = all_preds[i].sum(axis=0) / all_preds[i].shape[0]
        true_leads = LeadNamesToVector(moscow_leads[test_record_names[i].split("/")[-1]])
        pd.DataFrame([LEAD_NAMES, leads_dist, true_leads]).transpose().to_csv(path+"/Split"+str(split_index)+"/LeadsLocalization"+test_record_names[i].split("/")[1]+".txt", 
                                                                  header=None, index=None, sep=' ', mode='w') 

def CreateFolder(path, n_splits):
    try:  
        os.mkdir(path)  
    except OSError as error:
        True

    for i in range(n_splits):
        try:
            os.mkdir(path + "/Split" + str(i))
        except OSError as error:
            continue

### WRITE EDF ###

def CreateNewAnnotation(time_start, labels, freq): 
    freq = 1/freq
    i = 0
    label_starts = [time_start]
    label_lens = [-1]
    desc = ["StartPredictionTime"]
    while i < len(labels):
        if labels[i] == 1:
            desc.append("ModelPrediction")
            label_starts.append(time_start + i*freq)
            cur_start = i
            while i < len(labels) and labels[i] == 1:
                i += 1
            label_lens.append((i - cur_start) * freq)
        i += 1
    label_starts += [time_start + i*freq]
    label_lens += [-1]
    desc += ["EndPredictionTime"]

    return np.array(label_starts), np.array(label_lens), np.array(desc)


def WriteEDF(predictions, annots, record_names, path):
    freq = 80000 / 401
    for i in range(len(record_names)):
        time_start = annots[i][0][0]
        #print(time_start)
        preds_annotations = CreateNewAnnotation(time_start, predictions[i], freq)
        data = mne.io.read_raw_edf("data/" + record_names[i])

        preds_annotations = list(preds_annotations)
        preds_annotations[1] = np.clip(preds_annotations[1], a_min=0, a_max = None)
        #print(np.isnan(preds_annotations[1]).any())

        old_annot = np.array([[data.annotations[i]["onset"], data.annotations[i]["duration"], data.annotations[i]["description"]] 
                      for i in range(len(data.annotations))])
        
        full_annot = np.concatenate([np.array(preds_annotations), old_annot.T], axis=1)
        annotations = mne.Annotations(np.array(full_annot)[0], np.array(full_annot)[1], np.array(full_annot)[2])
        #print(annotations)
        data.set_annotations(annotations)
        
        data.export(path + "/Preds_" + record_names[i].split("/")[1], overwrite=True)
        data.close()


def WriteEDFCrossValid(predictions, split_index, annots, path, N):
    train_indices = np.ones(N) #number of records
    
    if N %2 == 1 and split_index == N // 2 + 1:
        train_indices[-1] = 0
    else:
        train_indices[2*split_index: 2*split_index+2] = 0
        
    train_record_names, test_record_names = GetTrainTestByIdxs(record_names, train_indices)

    train_annots, test_annots = GetTrainTestByIdxs(annots, train_indices)
    WriteEDF(predictions, test_annots, test_record_names, path + "/Split" + str(split_index))
    



### RUN EXPEREMENT ###

def CrossValidationExperement(record_names, moscow_leads, records, annots, times, labels, path, shuffle_leads=False, last_epoch=False, low_freq=True, 
                              is_train=True, write_edf=False, area_SWI=False, sneos=None, mcs=None, localization=False, win_len=500):
    N = len(records)
    n_splits = N // 2
    if N % 2 == 1:
        n_splits += 1
    CreateFolder(path, n_splits)
    for i in range(n_splits):
        train_data, train_labels, test_data, test_labels, train_records_names, test_records_names = GetTrainTestSplit(record_names, 
                                                                                                                      records, 
                                                                                                                      annots, times, 
                                                                                                                      labels, i, 
                                                                                                                      shuffle_leads,
                                                                                                                      N, sneos, mcs)
        train_samples = []
        for j in range(len(train_data)):
            if not localization:    
                train_samples += CreateSamples(train_data[j][0], train_labels[j], sneo=train_data[j][1], mc=train_data[j][2])
            else:
                train_samples += CreateInoutSeqLocalization(train_data[j][0], train_labels[j], 
                                              LeadNamesToVector(moscow_leads[train_records_names[j].split("/")[-1]]), win_len)
                

        test_samples = []
        for j in range(len(test_data)):
            if not localization:
                test_samples += CreateSamples(test_data[j][0], test_labels[j], sneo=test_data[j][1], mc=test_data[j][2])
            else:
                test_samples += CreateInoutSeqLocalization(test_data[j][0], test_labels[j], 
                                              LeadNamesToVector(moscow_leads[test_records_names[j].split("/")[-1]]), win_len)
            
        train_dataloader = DataLoader(train_samples, batch_size=32, shuffle=True, drop_last=True) # or train_samples for 4000 or new_train_samples for 100
        test_dataloader = DataLoader(test_samples, batch_size=32, shuffle=False, drop_last=True) # often changes, may be add to parameter
        model = UNET_1D(20,128,7,3,1)
        if localization:
            model = UNET_1D_Localization(window_len=win_len, input_dim=20, layer_n=128, kernel_size=7, n_down_layers=3, depth=2)
        if sneos is not None:    
            model = UNET_1D(20,128,7,3,3) #(input_dim, hidden_layer, kernel_size, depth)
        model = model.to("cuda")
        if is_train:
            Train(model, train_dataloader, test_dataloader, i, path, localization)
        if not localization:
            LogResults(model, test_data, test_labels, i, path, last_epoch, low_freq, write_edf, annots, area_SWI, N)      
        else:
            LogResultsLocalization(model, test_data, test_labels, test_records_names, moscow_leads, i, path, win_len)  

def CreateFakeAnnots(times):
    return [[[0], [time[-1]]] for time in times]
            
def AllRecordsPredictionNewRecords(records, annots, times, labels, 
                                   test_record_names, test_records, test_annots, test_times,
                                   path, shuffle_leads=False, is_train=True, sneos=None, mcs=None):
    N = len(records)
    n_splits = 1
    CreateFolder(path, n_splits)
    for i in range(n_splits):
        fake_labels = [np.zeros(len(record[0])) for record in test_records]
        train_data, train_labels, test_data, test_labels = WrapTrainTestInput(records, annots, times, [None]*3, labels.copy(), 
                                                                 test_records, test_annots, test_times, [None] * 3, fake_labels, shuffle_leads) #not sneos yet                                                                                  
        train_samples = []
        for j in range(len(train_data)):   
            train_samples += CreateSamples(train_data[j][0], train_labels[j], sneo=train_data[j][1], mc=train_data[j][2])
           
        test_samples = []
        for j in range(len(test_data)):
            test_samples += CreateSamples(test_data[j][0], test_labels[j], sneo=None, mc=None)
            
        train_dataloader = DataLoader(train_samples, batch_size=32, shuffle=True, drop_last=True) # or train_samples for 4000 or new_train_samples for 100
        test_dataloader = DataLoader(test_samples, batch_size=1, shuffle=False, drop_last=True) # often changes, may be add to parameter
        model = UNET_1D(20,128,7,3,1)
        if sneos is not None:    
            model = UNET_1D(20,128,7,3,3) #(input_dim, hidden_layer, kernel_size, depth)
        model = model.to("cuda")
        if is_train:
            Train(model, train_dataloader, test_dataloader, i, path, localization=False)
            torch.save(model.state_dict(), path + "/Split" + str(i) +"/Unet1d")
        else:
            model.load_state_dict(torch.load(path + "/Split" + str(i) +"/Unet1d"))
            
        test_predicts = CollectingPreds(model, test_data)
        for j in range(len(test_predicts)):
            PostProcessing(test_predicts[j], 20)
        WriteEDF(test_predicts, test_annots, test_record_names, path + "/Split" + str(i))