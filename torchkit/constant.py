import torch.nn as nn
from utils.conv_lstm_hzzone import ConvLSTM
from utils.equi_conv_lstm_hzzone import ConvLSTM as EqConvLSTM

LSTM_name = "lstm"
GRU_name = "gru"
ConvLSTM_name = "conv-lstm"
EqConvLSTM_name = "equi-conv-lstm"
RNNs = {
    LSTM_name: nn.LSTM,
    GRU_name: nn.GRU,
    ConvLSTM_name: ConvLSTM,
    EqConvLSTM_name: EqConvLSTM
}
