import torch
import torch.nn as nn

def weights_init_norm(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1: 
        nn.init.norm_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.norm_(m.weight)

def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1: 
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform_(m.weight)

def to_device(obj, device):
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(list(obj), device))
    elif isinstance(obj, dict):
        retval = dict()
        for key, value in obj.items():
            retval[to_device(key, device)] = to_device(value, device)
        return retval 
    elif hasattr(obj, "to"): 
        return obj.to(device)
    else: 
        return obj

##
# MuLaw
eps = 0e-6
def mulaw_encode(x, mu):
    # [-1.0, 1.0] to [0, mu)
    return ((x + 1.0) * (mu / 2) - eps)

def mulaw_decode(q, mu):
    # [0, mu) to [-1.0, 1.0]
    x = ((q) / mu) * 2 - 1.
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
    return x
