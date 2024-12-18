import difflogic
import difflogic.functional as df
import torch
import torch.nn as nn
import numpy
import math


"""

-> matrice of size [out_channels, 2]
-> allows to choose which input channels to take for which out_channel (each out_channel has access to 2 input channels)
layer_id = out_channels_in_idx


-> per out_channel, input idx to choose (from 0 to 17 since we have rf*rf (with rf=3) per channel to choose from, which is 2)
idx = indices


x of shape [B, Cin, H, W]

-> pad x so that H and W dim stay the same size

===> out shape: [B, Cin, H + rf // 2, W + rf // 2]

-> unfold on dim 2 and 3 by receiptive field (with stride of 1)
xu = x.unfold(2, rf, stride).unfold(3,rf,stride)
===> out shape: [B, Cin, H, W, rf, rf]

-> flatten kernel dims
xu2 = xu.flatten(start_dim=4)
===> out shape: [B, Cin, H, W, rf**2]

-> transpose to flatten after
xu3 = xu2.transpose(2,1).transpose(2,3)
===> out shape: [B, H, W, Cin, rf**2]

-> concat in_channels per 2 into the out_channels and choose randomly 8 inputs per out_channel
f = torch.stack([torch.cat((xu3[:,:,:,layer_id[i][0],:], xu3[:,:,:,layer_id[i][1],:]), axis=-1)[..., idx[i]] for i in range(out_channels)], dim=-2)
===> out shape: [B, H, W, Cout, num_inp_per_tree]

-> flatten before sending to Logic Layer
ff = ft.flatten(start_dim=3)
===> out shape: [B, H, W, Cout * num_inp_per_tree]


===> shape after Logic Layer (3 logic layer, each of them divides input per 2): [B, H, W, Cout]

details:
while choosing the connection in the layer, it takes the pairs sequentially, i.e. (0,1),(2,3),(4,5),...,(n-1,n) with n = 8 * out_channel (as there is 8 input per channel)
doing that, the inputs between layers are not mixed


"""


class ConvolutionalLogicTree(nn.Module):
    def __init__(self, in_channels, out_channels, depth, receptive_field, grad_factor, device, stride=1, pad=True): 
        super().__init__()
        # general fields
        num_inputs_per_tree = 2**depth
        self.offset = receptive_field // 2
        self.out_channels = out_channels
        self.receptive_field = receptive_field
        self.stride = stride
        self.pad = pad

        # backbone layers
        lgn_dim = [2**depth // 2**i * out_channels for i in range(depth+1)]
        self.layers = nn.Sequential(*[
            LogicLayer_residual(in_dim=lgn_dim[i], out_dim=lgn_dim[i+1], device=device, grad_factor=grad_factor, connections='unique')
            for i in range(depth)
        ])

        # forward's auxiliary fields
        scale = math.ceil(out_channels / in_channels)
        assert 2 <= scale # else we need another condition
        tmp = torch.stack([torch.arange(in_channels) for _ in range(scale)])
        self.out_channels_in_idx = torch.stack([tmp.flatten(), tmp.T.flatten()], axis=-1)[:out_channels]
        #print(self.out_channels_in_idx)

        # indices per tree to select input, chosen randomly
        self.indices = torch.stack([torch.randperm((receptive_field**2)*2)[:num_inputs_per_tree] 
                                    for _ in range(out_channels)])

    def forward(self, x):
        #print(x.shape)
        assert x.ndim == 4, x.ndim
        # in: [B, Cin, H, W]
        #print("[B, Cin, H, W]")
        bs = x.shape[0]
        if self.pad:
            h = x.shape[2]
            w = x.shape[3]
        else:
            h = x.shape[2] - 2 * self.offset
            w = x.shape[3] - 2 * self.offset

        # pad
        # out: [B, Cin, H+offset, W+offset]
        if self.pad:
            x = nn.functional.pad(x, (self.offset,self.offset,self.offset,self.offset), 'constant', 0)
        #print(x.shape)
        #print("[B, Cin, H+offset, W+offset]")

        # unfold
        # out: [B, Cin, H, W, rf, rf]
        x = x.unfold(2, self.receptive_field, self.stride).unfold(3, self.receptive_field, self.stride)
        #print(x.shape)
        #print("[B, Cin, H, W, rf, rf]")

        # flatten & transpose
        # out: [B, H, W, Cin, rf*rf]
        x = x.flatten(start_dim=4)
        x = x.transpose(2,1).transpose(2,3)
        #print(x.shape)
        #print("[B, H, W, Cin, rf*rf]")

        # concat -> select -> stack -> transpose -> flatten 
        # out: [B, H, W, Cout * 2**depth]
        x = torch.stack([torch.cat(( # cat two input channels per out_channel
            x[:,:,:,self.out_channels_in_idx[i][0],:], # select first input channel for out_channel i
            x[:,:,:,self.out_channels_in_idx[i][1],:]), axis=-1)[..., self.indices[i]] # select second
            for i in range(self.out_channels)], dim=-2) # stack all on new dimension: out_channel
        x = x.flatten(start_dim=3)
        #print(x.shape)
        #print("[B, H, W, Cout * 2**depth]")

        # flatten B, H, W 
        # out: [B*H*W, Cout * 2**depth]
        x = x.flatten(end_dim=-2)
        #print(x.shape)
        #print("[B * H * W, Cout * 2**depth]")

        # ready for convolution

        out = self.layers(x)
        # out: [B * H * W, Cout]
        #print(out.shape)
        #print("[B * H * W, Cout]")
        out = out.reshape(bs, h, w, -1).transpose(-1,-2).transpose(1,2)
        # out: [B, Cout, H, W]
        #print(out.shape)
        #print("[B, Cout, H, W]")
        return out



class LogicLayer_residual(difflogic.LogicLayer):
    def __init__(self, *args, init='residual', **kwargs):
        super().__init__(*args,**kwargs)
        assert init in ['residual', 'random']
        if init == 'residual':
            w = torch.tensor([[0.067] * 3 + [8.995] + [0.067] * 12 for _ in range(self.out_dim)], device=self.device)
            self.weights = torch.nn.parameter.Parameter(w)

    def forward(self, x):
        #print(x.shape)
        #print(type(x))
        return super().forward(x)



class LogicTreeNet(nn.Module):
    def __init__(self, in_channels, k, num_classes,d, rf, grad, tau, device):
        super().__init__()

        self.conv1 = ConvolutionalLogicTree(in_channels, k, d, 5, grad, device, pad=False)
        self.conv2 = ConvolutionalLogicTree(k, k*3, d, rf, grad, device)
        self.conv3 = ConvolutionalLogicTree(k*3, k*9, d, rf, grad, device)
        #self.conv4 = ConvolutionalLogicTree(k*16, k*32, d, rf, grad, device)

        self.or_pooling = nn.MaxPool2d(2)

        self.lg1 = difflogic.LogicLayer(81*k, 1280*2*k, device, grad)
        self.lg2 = difflogic.LogicLayer(1280*k*2, 640*k*2, device, grad)
        self.lg3 = difflogic.LogicLayer(640*k*2, 320*k*2, device, grad)

        self.group_sum = difflogic.GroupSum(num_classes, tau)

    def forward(self, x):

        x = self.or_pooling(self.conv1(x))
        x = self.or_pooling(self.conv2(x))
        x = self.or_pooling(self.conv3(x))
        #x = self.or_pooling(self.conv4(x))

        x = x.flatten(start_dim=1)

        x = self.lg1(x)
        x = self.lg2(x)
        x = self.lg3(x)

        out = self.group_sum(x)

        return out