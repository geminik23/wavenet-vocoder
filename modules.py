import torch
import torch.nn as nn

from utils import mulaw_encode

class MuLawEncoder(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        
    def forward(self, input):
        return mulaw_encode(input, self.mu)

class CasualConv1d(nn.Module):
    """
    Casual 1D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, exclude_last=False, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.E = exclude_last
        
        self.padding = (kernel_size - 1)*dilation + exclude_last*1
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation, **kwargs)
    
        
    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, 0))
        out = self.conv1d(x)
        if self.E:
            return out[..., :-1] # exclude last sample
        else:
            return out

class GatedResidualBlock(nn.Module):
    """
    Residual Block 
    """
    def __init__(self, residual_channels, skip_channels, condition_channels, causal_kernel=2, causal_dilation=1, causal_exclude_last=False):
        super().__init__()

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.condition_channels = condition_channels
        self.causal_kernel = causal_kernel
        self.causal_dilation = causal_dilation
        self.causal_exclude_last = causal_exclude_last 
        self.hidden_channels = residual_channels 

        self.causal_conv_f = CasualConv1d(residual_channels, self.hidden_channels, causal_kernel, causal_dilation, exclude_last=causal_exclude_last)
        self.causal_conv_g = CasualConv1d(residual_channels, self.hidden_channels, causal_kernel, causal_dilation, exclude_last=causal_exclude_last)
        self.conditional_conv_f = nn.Conv1d(condition_channels, self.hidden_channels, kernel_size=1) if condition_channels is not None else None
        self.conditional_conv_g = nn.Conv1d(condition_channels, self.hidden_channels, kernel_size=1) if condition_channels is not None else None

        self.skip_conv = nn.Conv1d(self.hidden_channels, skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(self.hidden_channels, residual_channels, kernel_size=1)

        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, condition):
        """
        B: batch_size, R: residual_channel, T: time series
        C: condition_channel (assume C is n_mels), S: skip_channel
        input : [B, R, T]
        condition : [B, C, T] 
        """

        # -> [B, 2*R, T]
        gates = torch.mul(
            self.tanh(self.causal_conv_f(input) + self.conditional_conv_f(condition)), 
            self.sigm(self.causal_conv_g(input) + self.conditional_conv_g(condition))
            )
        
        # [B, 2*R, T] -> [B, S, T]
        skips = self.skip_conv(gates)

        # [B, 2*R, T] -> [B, R, T]
        residuals = self.residual_conv(gates) + input
        return skips, residuals

