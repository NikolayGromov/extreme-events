import mne
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.signal import butter, filtfilt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


#Data collection
def ReadSignal(file_name): 

    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((20, f.getNSamples()[0])) #or n
    
    if n == 22:
        for i in np.arange(19):
            sigbufs[i, :] = f.readSignal(i)
        sigbufs[19, :] = f.readSignal(21)
    elif n == 23:
        for i in np.arange(19):
            sigbufs[i, :] = f.readSignal(i)
        sigbufs[19, :] = f.readSignal(20)
    else:
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)

    time = [1/f.samplefrequency(0) * i for i in range(len(sigbufs[0]))]

    annotations = f.readAnnotations()  


    new_annot = [(annotations[0][i], annotations[1][i], annotations[2][i])  
                 for i in range(len(annotations[0])) 
                                if (annotations[1][i] > 0) and (annotations[2][i] in ['?', 'Ð²Ñ\x81', 'Ð²Ñ\x81?', 'Ð²Ñ\x81-'])]
    f.close()
    return sigbufs, new_annot, time, f.samplefrequency(0)

def ParseSplits(exp_name, record_names):
    # Создаем словарь для хранения соответствий: record_name -> путь к SEED
    record_to_split = {}

    # 1. Сканируем все подпапки Split*
    for split_dir in os.listdir(exp_name):
        if not split_dir.startswith("Split"):
            continue
            
        split_path = os.path.join(exp_name, split_dir)
        metrics_file = os.path.join(split_path, "Metrics.txt")
        
        # Читаем содержимое Metrics.txt
        with open(metrics_file, 'r') as f:
            content = f.read()
        
        # 2. Ищем соответствие с record_names
        for record_name in record_names:
            if record_name in content:
                record_to_split[record_name] = split_dir
                break  # Нашли соответствие, переходим к следующему Split
    return record_to_split


def broad_filter(signal, fs, lowcut=0.1, highcut=35):
    """Returns filtered signal sampled at fs Hz, with a [lowcut, highcut] Hz
    bandpass."""
    # Generate butter bandpass of order 3.
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(3, (low, high), btype='band')
    # Apply filter to the signal with zero-phase.
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def NormalizeAndClip(data):
    for i in tqdm(range(len(data))):
        signal = data[i]
        means = signal.mean(axis=1)[..., None]
        stds = signal.std(axis=1)[..., None]
        signal = np.clip((signal - means) / stds, a_min=-10, a_max=10)
        data[i] = signal

#Characteristics

def CollectAmplitude(x):
    amplitudes = np.max(x[:19], axis=1)
    return amplitudes.mean()

def CollectFreq(x):
    centered_x = x[:19] - np.mean(x[:19], axis=1)[..., None]
    zero_cross_num = ((centered_x[:, :-1] * centered_x[:, 1:]) < 0).sum(axis=1)
    return zero_cross_num.mean() / x.shape[1]

def CollectCharacteristics(x, labels):  
    lens = []
    ampls = []
    freqs = []
    
    c = 0
    in_event = False
    for i in range(len(labels)):
        if labels[i] == 1:
            if not in_event:
                idx_start = i
            in_event = True
            c += 1
        elif in_event:
            ampls.append(CollectAmplitude(x[:, idx_start:i]))
            freqs.append(CollectFreq(x[:, idx_start:i]))
            lens.append(c)
            c = 0
            in_event = False
                
    return np.mean(lens), np.mean(ampls), np.mean(freqs), len(lens)   

def CreateSamplesCharacteristics(x, true_labels, pred_labels, window=10, sr=256): #window in minutes
    sample_window = window * 60 * sr
    inout_seq_labels = []
    inout_seq_characteristics_true = []
    inout_seq_characteristics_pred = []
    for i in range(0, x.shape[1] - sample_window, sample_window):
        if (true_labels[i:i + sample_window]).max() == 1:
            train_seq = x[:, i:i+sample_window].numpy()
            true_label = true_labels[i:i+sample_window].numpy()
            pred_label = pred_labels[i:i+sample_window]
            duration_true, amplitude_true, freq_true, ss_number_true = CollectCharacteristics(train_seq, true_label)
            duration_pred, amplitude_pred, freq_pred, ss_number_pred = CollectCharacteristics(train_seq, pred_label)
            
            inout_seq_characteristics_true.append(np.array([duration_true, amplitude_true, freq_true, ss_number_true]))
            inout_seq_characteristics_pred.append(np.array([duration_pred, amplitude_pred, freq_pred, ss_number_pred]))
    return inout_seq_characteristics_true, inout_seq_characteristics_pred


#Networks

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
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
    def __init__(self ,input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5, padding=8)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25, padding=8)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125, padding=8)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest') #for 4000 it is 5 and for 100 is 4
        
        self.outcov = nn.Conv1d(self.layer_n, 2, kernel_size=self.kernel_size, stride=1,padding = 3)
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        #print("x", x.size())
        pool_x1 = self.AvgPool1D1(x)
        #print("pool_x1", pool_x1.size())
        
        pool_x2 = self.AvgPool1D2(x)
        #print("pool_x2", pool_x2.size())
        
        pool_x3 = self.AvgPool1D3(x)
        #print("pool_x3", pool_x3.size())
        
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        #print("out_0", out_0.size())
        out_1 = self.layer2(out_0)        
        #print("out_1", out_1.size())
        
        
        x = torch.cat([out_1,pool_x1],1)

        #print("x", x.size())
        out_2 = self.layer3(x)
        #print("out_2", out_2.size())
        
        x = torch.cat([out_2,pool_x2],1)
        #print("x", x.size())
        x = self.layer4(x)
        #print("x", x.size())
        
        
        #############Decoder####################
        
        up = self.upsample1(x)
        #print("up", up.size())
        
        up = torch.cat([up,out_2],1)
        #print("up", up.size())
        
        up = self.cbr_up1(up)
        #print("up", up.size())
        
        up = self.upsample(up)
        #print("up", up.size())
        
        up = torch.cat([up,out_1],1)
        #print("up", up.size())
        
        up = self.cbr_up2(up)
        #print("up", up.size())
        
        
        up = self.upsample(up)
        #print("up", up.size())
        
        up = torch.cat([up,out_0],1)
        #print("up", up.size())
        
        up = self.cbr_up3(up)
        #print("up", up.size())
        
        out = self.outcov(up)
        #print("out", out.size())
        
        #out = nn.functional.softmax(out,dim=2)
        
        return out

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
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        
        x = self.bn2(x)
        x = self.relu(x)

        x = self.AvgPool1D(x)
        
        
        x = self.mdb1(x)
        

        x = self.conv3(x)
        
        
        x = self.bn3(x)
        x = self.relu(x)

        x = self.AvgPool1D(x)
        
        
        x = self.mdb2(x)
        
        

        x = self.AvgPool1D(x)
        

        x = x[:, :, 65:-64]
        

        x = self.dropout1(x)

        lstm_out, (h_n, c_n) = self.lstm1(x.transpose(2, 1))

        

        x = lstm_out[:, :, :self.N1] + lstm_out[:, :, self.N1:]
        x = self.dropout2(x)

        

        lstm_out, (h_n, c_n) = self.lstm2(x)

        

        x = lstm_out[:, :, :self.N1] + lstm_out[:, :, self.N1:]
        x = self.dropout2(x)
        
        x = x.transpose(2, 1)
        x = self.dropout2(x)
        

        x = self.classifier1(x)
        
        
        x = self.bn5(x)
        x = self.relu(x)

        x = self.classifier2(x)    
        

        x =self.upsample(x)    
        
    
        return x
        
        
#Colletcting Preds

RECEPTIVE_FIELD = 4000
OVERLAP = 520
def CollectingPreds(model, test_data, model_name):

    model.eval()
    model.cpu()
    all_preds = []
    
    overlap = 0
    if model_name == "SEED":
        overlap=520
    for i in range(len(test_data)):
        record_preds = []
        
        for idx in tqdm(range(overlap, test_data[i].size()[1]- RECEPTIVE_FIELD - overlap, RECEPTIVE_FIELD)):

            train_seq = test_data[i][:, idx-overlap:idx+RECEPTIVE_FIELD+overlap][None, ...]
                  
            out = model(train_seq)
            m = nn.Softmax(dim=1)
            out = m(out)
            
            preds = np.argmax(out.detach().cpu().numpy(), axis=1)
            record_preds.append(preds)
        shapes = np.array(record_preds).shape
        record_preds = np.array(record_preds).reshape(shapes[0] * shapes[1] * shapes[2])
        all_preds.append(record_preds)
    return all_preds

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
def PostProcessing(predictions, threshold):
    MergeClose(predictions, threshold)
    DeleteShortEvents(predictions, threshold)


def DownloadChatsPredictions(char_exp_name, record_names, records, splits_dict, path):
    all_answers = []
    all_preds_chars = []
    for i in range(len(records)):
        split_name = splits_dict[record_names[i]]
        all_answers.append(np.load(path + "CrossValidationResults/" + char_exp_name + "/" + split_name + "/Answers.npy"))
        all_preds_chars.append(np.load(path + "CrossValidationResults/" + char_exp_name + "/" + split_name +  "/Preds.npy"))
    all_answers = np.concatenate(all_answers, axis=1)
    all_preds_chars = np.concatenate(all_preds_chars, axis=1)  
    all_answers[2] *= 256
    all_preds_chars[2] *= 256  

    abs_difs = []
    for i in range(4):
        abs_difs.append(np.abs(all_preds_chars[i] - all_answers[i]))
    return all_answers, all_preds_chars, abs_difs

def CharsFromPredictions(epi_data, epi_labels, segm_model_predictions, overlap):
    test_samples_char_true = []
    test_samples_char_pred = []

    for i in range(len(epi_data)):
        inout_seq_char_true, inout_seq_char_pred = CreateSamplesCharacteristics(epi_data[i][:, overlap:], epi_labels[i][1][overlap:],
                                                                                segm_model_predictions[i], window=2, sr=199)
        test_samples_char_true += inout_seq_char_true
        test_samples_char_pred += inout_seq_char_pred    

    test_samples_char_true = np.array(test_samples_char_true)
    test_samples_char_pred = np.array(test_samples_char_pred)

    test_samples_char_true[:, 2] *= 256
    test_samples_char_pred[:, 2] *= 256

    epi_dif = np.abs(test_samples_char_true - test_samples_char_pred)

    return test_samples_char_true, test_samples_char_pred, epi_dif
