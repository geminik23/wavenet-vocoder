import torch
import torch.nn as nn
import numpy as np
import unittest
from modules import CasualConv1d, GatedResidualBlock
from model import WaveNet_Mel2Raw
from config import Config
import math



class WaveNetTestCase(unittest.TestCase):

    def test_to_device(self):
        # test this when 'cuda' is available
        if not torch.cuda.is_available():
            self.assertTrue(False, 'cuda is not available')

        from utils import to_device
        cuda = torch.device('cuda:0')
        cpu = torch.device('cpu')

        def is_all(tensors:list, device):
            return np.all([t.device == device for t in tensors])

        a = torch.randn((2, 1, 10)).to(cpu)
        b = torch.randn((2, 1, 10)).to(cpu)
        c = torch.randn((2, 1, 10)).to(cpu)
        self.assertTrue(is_all([a,b,c], cpu))

        # list 
        l = [a,b,c]
        l = to_device(l, cuda)
        self.assertTrue(is_all(l, cuda))

        # tuple
        t = (a,b,c)
        t = to_device(t, cpu)
        self.assertTrue(is_all(list(t), cpu))
        
        # dict
        d = {'a':a, 'b':b, 'c':c}
        d = to_device(d, cuda)
        self.assertTrue(is_all(list(d.values()), cuda))
    

    def test_mulaw(self):
        from utils import mulaw_decode, mulaw_encode
        Q = 256
        batch_size = 2
        length = 2000

        # [-1. ~ 1.]
        x = torch.rand((batch_size, length))*2-1
        self.assertTrue(x.min()>=-1. and x.max()<=1.)
        mu = torch.tensor([Q-1], dtype=torch.float32)
        q = mulaw_encode(x, mu)
        self.assertTrue(q.min()>=0 and q.max()<Q)
        y = mulaw_decode(q, mu)
        self.assertTrue(y.min()>=-1. and y.max()<=1.)


    def test_gatedreisualblock(self):
        residual_channels = 64
        skip_channels = 128
        n_mels = 80
        batch_size=32
        length = 1000
        
        input, condition = torch.randn((batch_size, residual_channels, length )),torch.randn((batch_size, n_mels, length))
        net = GatedResidualBlock(residual_channels, skip_channels, n_mels)
        skip, residual = net(input, condition)


        self.assertEqual(torch.Size((batch_size, skip_channels, length)), skip.shape)
        # residual will be input of nexst block, so same shape
        self.assertEqual(input.shape, residual.shape)

        
    def test_causalconv1d(self):
        net1 = CasualConv1d(1, 1, 2, 1, True)
        nn.init.constant_(net1.conv1d.weight, 1)
        nn.init.constant_(net1.conv1d.bias, 0)

        net2 = CasualConv1d(1, 1, 2, 1, False)
        nn.init.constant_(net2.conv1d.weight, 1)
        nn.init.constant_(net2.conv1d.bias, 0)

        x = torch.tensor([1,1,1,1]).view(1, -1).float()
        y1 = net1(x)
        y2 = net2(x)

        # two tensors are same shape
        self.assertTrue(x.shape, y1.shape)
        self.assertTrue(x.shape, y2.shape)

        # expected output
        self.assertTrue(torch.equal(y1, torch.tensor([[0., 1., 2., 2.]])))
        self.assertTrue(torch.equal(y2, torch.tensor([[1., 2., 2., 2.]])))

    
    def test_wavenet_model_forward(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = Config()
        model = WaveNet_Mel2Raw.create_with(config).to(device)

        #[B, T]
        n_mels = config.n_mels
        batch_size = 2
        length = 5000
        frame_length = math.ceil(length/config.hop_size)

        x = torch.rand((batch_size, length)).to(device)*2. - 1.
        x = model.mulaw_encode(x)
        self.assertEqual(x.dtype, torch.long)

        mels = torch.randn((batch_size, n_mels, frame_length)).to(device)
        y = model((x, mels))
        
        target_shape = torch.Size([batch_size, config.num_class, length])

        self.assertEqual(y.shape[:2], target_shape[:2])
        self.assertTrue(target_shape[2] in range(y.shape[2]-config.hop_size, y.shape[2]+config.hop_size))

    @unittest.SkipTest
    def test_wavenet_inference(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = Config()
        model = WaveNet_Mel2Raw.create_with(config).to(device)

        #[B, T]
        n_mels = config.n_mels
        batch_size = 2
        length = 2000
        frame_length = (length//config.hop_size) + 1

        mels = torch.randn((batch_size, n_mels, frame_length)).to(device)

        y = model.inference(mels, device)
        target_shape = torch.Size([batch_size, length])
        self.assertTrue(target_shape[1] in range(y.shape[1]-config.hop_size, y.shape[1]+config.hop_size))

        
if __name__ == '__main__':
    unittest.main(verbosity=2)