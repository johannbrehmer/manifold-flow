import torch

from torch import nn
from torch.nn import functional as F

from manifold_flow.utils import various


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class AttentionBlock(nn.Module):
    def __init__(self, channels, which_conv=nn.Conv2d, heads=8):
        super(AttentionBlock, self).__init__()
        # Channel multiplier
        self.channels = channels
        self.which_conv = which_conv
        self.heads = heads
        self.theta = self.which_conv(self.channels, self.channels // heads, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.channels, self.channels // heads, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.channels, self.channels // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.channels // 2, self.channels, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, inputs, y=None):
        # Apply convs
        theta = self.theta(inputs)
        phi = F.max_pool2d(self.phi(inputs), [2, 2])
        g = F.max_pool2d(self.g(inputs), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.channels // self.heads, inputs.shape[2] * inputs.shape[3])
        phi = phi.view(-1, self.channels // self.heads, inputs.shape[2] * inputs.shape[3] // 4)
        g = g.view(-1, self.channels // 2, inputs.shape[2] * inputs.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.channels // 2, inputs.shape[2], inputs.shape[3]))
        outputs = self.gamma * o + inputs
        return outputs


class ConvAttentionNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_blocks):
        super().__init__()
        self.initial_layer = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.attention_blocks = nn.ModuleList([AttentionBlock(channels=hidden_channels, which_conv=nn.Conv2d, heads=8) for _ in range(num_blocks)])
        # if use_batch_norm:
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm2d(num_features=hidden_channels)])
        self.final_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for attention, batch_norm in zip(self.attention_blocks, self.batch_norm_layers):
            temps = attention(temps)
            temps = batch_norm(temps)
        outputs = self.final_layer(temps)
        return outputs


def main():
    batch_size, channels, height, width = 100, 12, 64, 64
    inputs = torch.rand(batch_size, channels, height, width)
    net = ConvAttentionNet(in_channels=channels, out_channels=2 * channels, hidden_channels=32, num_blocks=4)
    print(various.get_num_parameters(net))
    outputs = net(inputs)
    print(outputs.shape)


if __name__ == "__main__":
    main()
