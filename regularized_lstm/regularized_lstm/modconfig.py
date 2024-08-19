from dataclasses import dataclass


@dataclass
class RegularizedLSTMConfig:
    d_hidden = 512
    d_memcell = 256
    d_trans = 128
    MODEL_M = 20
    MODEL_N = 20