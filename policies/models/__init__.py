from .policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
from .policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_Separate_RNN
from enum import Enum


AGENT_CLASSES = {
    "Policy_MLP": Policy_MLP,
    "Policy_Separate_RNN": Policy_Separate_RNN,
}


class AGENT_ARCHS(str, Enum):
    # inherit from str to allow comparison with str
    Markov = Policy_MLP.ARCH
    Memory = Policy_Separate_RNN.ARCH
