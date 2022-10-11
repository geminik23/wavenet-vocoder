import torch
import torch.nn as nn
from utils import mulaw_decode, mulaw_encode
from modules import GatedResidualBlock
from config import Config



class WaveNet_Mel2Raw(nn.Module):
    @classmethod
    def create_with(cls, config:Config):
        return cls(config.n_mels, config.num_class, config.hop_size, config.residual_channels, config.skip_channels, config.upsample_kernel, config.causal_kernel, config.num_cycle_block, config.max_dilation)

    def __init__(self, n_mels, num_class, hopsize, residual_channels, skip_channels, upsample_kernel_size, causal_kernel, n_cycle_block, max_dilation=9):
        super().__init__()
        # residual = input_conv(input) 
        # condition = mel_conv(condition)
        # skips, residual = residual_blocks( residual, condition) ...
        # y = post_convs(skips)

        self.n_mels = n_mels
        self.num_class = num_class
        self.hopsize = hopsize
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.upsample_kernel_size = upsample_kernel_size
        self.causal_kernel = causal_kernel
        self.n_cycle_block = n_cycle_block
        self.max_dilation = max_dilation

        dilations = [2**i for i in range(max_dilation+1)]
        blocks = []

        mu = nn.Parameter(torch.tensor([self.num_class-1], dtype=torch.float), requires_grad=False)
        self.register_buffer("mu", mu)

        self.input_conv = nn.Conv1d(1, residual_channels, kernel_size=1)

        # in mels tensor,  frames -> audio length... [N_MELS, FRAME] -> [N_MELS, AUDIO_LENGTH]
        # padding = upsample_kernel_size + hopsize //2
        padding = upsample_kernel_size // 2
        self.mel_conv= nn.ConvTranspose1d(n_mels, n_mels, kernel_size=upsample_kernel_size, stride=hopsize, padding=padding)

        for n in range(n_cycle_block):
            for i, d in enumerate(dilations):
                blocks.append(GatedResidualBlock(residual_channels, skip_channels, n_mels, causal_kernel, d, i == 0 and n==0))
                pass
            pass

        self.residual_blocks = nn.ModuleList(blocks)
        
        self.post_convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, num_class, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_class, num_class, kernel_size=1),
        )
        self.receptive_field_size = self.get_receptive_field_size()

    def mulaw_encode(self, x):
        return mulaw_encode(x, self.mu).long()

    def get_receptive_field_size(self):
        ## this is only for kernel=2
        # receptive field size : 1, 2, 4, 8 -> 16 = 2^(max_dilation=4)
        # + include the num cycle of blocks  (2^max_dilation - 1)*n_cycle_block + 1
        # this is for inference
        return (2 ** self.max_dilation - 1) * self.n_cycle_block + 1

    def _f(self, input, condition):
        l = min(input.size(-1), condition.size(-1))
        x, condition = input[..., :l], condition[..., :l]
        # [B, T] -> [B, 1, T] # melspec: [B, n_mels, frame_length] -> [B, n_mels, out_length(~audio length)]
        x = x.unsqueeze(1)

        # [B, 1, T] -> [B, R, T]
        skips, residual= [], self.input_conv(x)
        for l in self.residual_blocks:
            # residuals : [B, R, T]
            # skip : [B, S, T]
            skip, residual = l(residual, condition)
            skips.append(skip)
        
        skip = torch.stack(skips, dim=0).sum(0) # [B, S, T]
        y = self.post_convs(skip) # [B, num_class, T]
        return y

    def forward(self, input):
        """
        B: batch_size, R: residual_channel, T: time series
        C: condition_channel (assume C is n_mels), F: Frame lengths, S: skip_channel
        input : (u-law encoded waveform, mels) [B, T], [B, C, F]

        RETURNS:
        output : [B, num_class, T]
        """
        # fit the T length
        assert len(input) == 2

        x, mels = input[0].float(), input[1]
        condition = self.mel_conv(mels) # local condition

        return self._f(x, condition)


    def inference(self, melspec, device='cpu'):
        """
        melspec : [B, n_mels, frame_length]
        """
        if not isinstance(melspec, torch.Tensor):
            melspec = torch.Tensor(melspec)
        if len(melspec.shape) == 2:
            melspec = melspec.unsqueeze(0)
        melspec = melspec.to(device)

        # mel to condition
        condition = self.mel_conv(melspec) # melspec: [B, n_mels, frame_length] -> [B, n_mels, out_length(~audio length)]


        # init with zeros the ouptut tensor
        raw_wav = torch.zeros((melspec.size(0), condition.size(-1)), dtype=torch.float).to(device)

        with torch.no_grad():
            for i in range(0, condition.shape[-1]):
                begin = max(i-self.receptive_field_size, 0)
                end = i+1
                x = raw_wav[:,begin:end]

                logits = self._f(x, condition[..., begin:end])
                p = nn.functional.softmax(logits, dim=1)

                # logits : [B, NUM_CLASS, receptive_field_size]
                # use only last one
                q = torch.argmax(p[...,-1], dim=1).unsqueeze(-1)
                y = mulaw_decode(q, self.mu) # [B]
                raw_wav[:, i] = y.squeeze(1)
        
        return raw_wav

    def load_checkpoint(self, cp_filepath):
        data = torch.load(cp_filepath)
        self.load_state_dict(data.get('model_state_dict'))

