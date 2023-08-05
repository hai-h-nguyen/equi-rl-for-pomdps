import torchkit.pytorch_utils as ptu
from utils.conv_lstm_hzzone import ConvLSTM
import torch
from utils import helpers as utl

input_size = 40
hidden_size = 128

rnn = ConvLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    aux_weights=True,
    batch_first=False,
    bias=False
)

utl.count_parameters(rnn)

input = torch.randn((1, 1, 40))
output, (h, c) = rnn(input, None)
output, (h, c) = rnn(input, (h, c))
output, (h, c) = rnn(input, (h, c))
output, (h, c) = rnn(input, (h, c))
# print(output.shape)
print(h.shape)
print(c.shape)

# input = torch.randn((201, 32, 40))
# output, (h, c) = rnn(input)
# print(output.shape)
# print(h.shape)
# print(c.shape)