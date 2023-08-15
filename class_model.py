import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import to_2tuple


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv3d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv_2D(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv_2D, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class se_block(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0, only_sp=0):
        super(se_block, self).__init__()
        self.fc1 = nn.Conv3d(in_channels=in_ch, out_channels=int(in_ch / rate), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels=int(in_ch / rate), out_channels=in_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        self.fc3 = nn.Conv3d(in_channels=in_ch, out_channels=int(in_ch / rate), kernel_size=1)
        self.fc4 = nn.Conv3d(in_channels=int(in_ch / rate), out_channels=in_ch, kernel_size=1)

        self.ch_use = only_ch
        self.ch_sp_use = only_sp

    def forward(self, x):
        x_in = x  # 4 16 64 64 64
        x = torch.mean(x.mean((3, 4), keepdim=True), 2, keepdim=True)  # 8 256 1 1
        x = self.fc1(x)  # 8 256 1 1 -> 8 64 1 1
        x = self.relu(x)  # 8 64 1 1 -> 8 64 1 1
        x = self.fc2(x)  # 8 64 1 1 -> 8 256 1 1
        if self.ch_use == 1:
            return x * x_in  # 注意力已赋予输入feature map
        elif self.ch_use == 0:
            x = x * x_in

        # 在这里再加上空间注意力
        s_in = x  # 8 256 12 16
        s = self.compress(x)  # 8 256 12 16 -> 8 2 12 16
        s = self.spatial(s)  # 8 2 12 16 -> 8 1 12 16
        if self.ch_sp_use == 1:
            return s  # 单独输出 注意力att
        elif self.ch_sp_use == 0:
            s = s * s_in

        # # 再加上上下文注意力  或许是上下文吧
        # c_in = s  # 8 256 12 16
        # c = self.fc3(s)  # 8 256 12 16 -> 8 64 12 16
        # c = self.relu(c)
        # c = self.fc4(c)  # 8 64 12 16 -> 8 256 12 16
        # c = self.sigmoid(c) * c_in  # 8 256 12 16 -> 8 256 12 16

        return s


class adaptive_depth_compression(nn.Module):
    def __init__(self, comp_rate_in, comp_rate_be, in_ch=[64, 32, 16, 8], comp_ch=1, kernel_size=1):
        super(adaptive_depth_compression, self).__init__()
        out_ch = [int(in_ch[0] * comp_rate_in), int(in_ch[1] * comp_rate_in), int(in_ch[2] * comp_rate_in), int(in_ch[3] * comp_rate_in)]  # [64 32 16 8] -> [48 24 12 6]
        be_ch = [out_ch[0], out_ch[0]+out_ch[1], out_ch[0]+out_ch[1]+out_ch[2], out_ch[0]+out_ch[1]+out_ch[2]+out_ch[3]]  # [64 32 16 8] -> [32 16 8]
        self.layer_0 = nn.Sequential(
            nn.Conv2d(in_ch[0], out_ch[0], kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[0]),
            nn.Conv2d(out_ch[0], out_ch[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[0])
        )
        self.trans_0 = nn.Sequential(
            nn.Conv2d(be_ch[0], in_ch[1], kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_ch[1])
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_ch[1], out_ch[1], kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[1]),
            nn.Conv2d(out_ch[1], out_ch[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[1])
        )
        self.trans_1 = nn.Sequential(
            nn.Conv2d(be_ch[1], in_ch[2], kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_ch[2])
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_ch[2], out_ch[2], kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[2]),
            nn.Conv2d(out_ch[2], out_ch[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[2])
        )
        self.trans_2 = nn.Sequential(
            nn.Conv2d(be_ch[2], in_ch[3], kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_ch[3])
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_ch[3], out_ch[3], kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[3]),
            nn.Conv2d(out_ch[3], out_ch[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch[3])
        )
        self.trans_3 = nn.Sequential(
            nn.Conv2d(be_ch[3], comp_ch, kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(comp_ch)
        )

    def forward(self, x):
        b, c, d, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        # x: 4 1 64 64 64 -> 4 1 64 64
        x_2d = x.view(b, -1, h, w)
        x_0 = self.layer_0(x_2d)  # 4 1 64 64 64 -> 4 1 48 64 64
        x_t_0 = self.trans_0(x_0)  # 4 1 48 64 64 -> 4 1 32 64 64

        x_1 = self.layer_1(x_t_0)  # 4 1 32 64 64 -> 4 1 24 64 64
        x_add_1 = torch.cat([x_0, x_1], dim=1)  # 4 1 48+24 64 64
        x_t_1 = self.trans_1(x_add_1)  # 4 1 72 64 64 -> 4 1 16 64 64

        x_2 = self.layer_2(x_t_1)  # 4 1 16 64 64 -> 4 1 12 64 64
        x_add_2 = torch.cat([x_0, x_1, x_2], dim=1)  # 4 1 48+24+12 64 64
        x_t_2 = self.trans_2(x_add_2)  # 4 1 84 64 64 -> 4 1 8 64 64

        x_3 = self.layer_3(x_t_2)  # 4 1 8 64 64 -> 4 1 6 64 64
        x_add_3 = torch.cat([x_0, x_1, x_2, x_3], dim=1)  # 4 1 48+24+12+6 64 64
        x_t_3 = self.trans_3(x_add_3)  # 4 1 90 64 64 -> 4 1 1 64 64

        x_2d_out = x_t_3

        return x_2d_out


class PFC_2D(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=7):
        super(PFC_2D, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))

    def forward(self, x_in):
        x_in = self.input_layer(x_in)
        residual = x_in
        x = self.depthwise(x_in)
        x = x + residual
        x = self.pointwise(x)
        return x


class PE_2D(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_chans=64, embed_dim=64, smaller=0):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=(patch_size[0] // 2)),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.proj_c = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True, groups=embed_dim),
            nn.ReLU(inplace=True)
        )
        self.fc0 = Conv_2D(embed_dim, embed_dim, 3, bn=True, relu=True)
        self.dwconv_1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True, groups=embed_dim),
            nn.ReLU(inplace=True)
        )
        self.use_small_conv = smaller
        self.dwconv_2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, stride=1, bias=True, groups=embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.turn_channel = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim * 2, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # overlap 编码
        x_pe = self.proj(x)  # 16 64 64 64 64 -> 16 64 32 32 32 # 重叠式编码
        # conv 编码
        x_pe_conv = self.proj_c(x) # 16 64 64 64 64 -> 16 64 32 32 32
        # fc_0
        x_PE = x_pe.flatten(2).transpose(1, 2)  # 16 64 32 32 32 ->16 64 32*32*32 -> 16 32*32*32 64
        x_PE = self.norm(x_PE)
        x_po = self.dwconv(x_pe).flatten(2).transpose(1, 2)  # 按照 unext 这里是加入位置编码
        x_0 = torch.transpose((x_PE + x_po), 1, 2).view(b, x_pe.shape[1], int(h / 2), int(w / 2)) # 16 32*32*32 64 -> 16 64 32 32 32
        x_0 = self.fc0(x_0)  # 16 24*32 64
        # fc_1
        x_1 = x_0  # torch.transpose(x_0, 1, 2).view(b, x_0.shape[2], int(h/2), int(w/2))
        if self.use_small_conv == 1:
            x_1_ = self.dwconv_2(torch.cat([x_1, x_pe_conv], dim=1))
            x_1_ = self.turn_channel(torch.cat([x_1_, x_pe], dim=1)).flatten(2).transpose(1, 2)
            x_out = x_1_ + x_PE
            x_out_reshape = torch.transpose(x_out, 1, 2).view(b, x_pe.shape[1], int(h / 2), int(w / 2))  # 16 32*32*32 64 -> 16 64 32 32 32
            return x_out_reshape
        else:
            x_1_ = self.dwconv_1(x_1)  # .flatten(2).transpose(1, 2) # 这里应该重新初始化一个dwconv吗
            x_1_ = self.turn_channel(torch.cat([x_1_, x_pe], dim=1)).flatten(2).transpose(1, 2)
            x_out = self.fc1(x_1_) + x_PE
            x_out_reshape = torch.transpose(x_out, 1, 2).view(b, x_pe.shape[1], int(h / 2), int(w / 2))  # 16 32*32*32 64 -> 16 64 32 32 32
            return x_out_reshape


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        # self.add_module('att0', se_block(bn_size * growth_rate, 0.5, 0, 0))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate
        # self.res_conv = Conv(inp_dim=num_input_features, out_dim=growth_rate, kernel_size=1, stride=1, bn=False, relu=False, bias=False)

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=5,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 5, 5),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 2, 2),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.pfc_2d = PFC_2D(in_ch=1, channels=64)
        self.pe_0_2d = PE_2D(patch_size=3, stride=2, in_chans=64, embed_dim=64, smaller=1)
        self.pe_1_2d = PE_2D(patch_size=3, stride=2, in_chans=64, embed_dim=128, smaller=1)
        self.pe_2_2d = PE_2D(patch_size=3, stride=2, in_chans=128, embed_dim=256, smaller=1)
        self.pe_3_2d = PE_2D(patch_size=3, stride=2, in_chans=256, embed_dim=512, smaller=1)
        self.drop_2d = nn.Dropout2d(0.3)
        self.avgpool_2d = nn.Sequential(
            Conv_2D(512, 1024, 1, 1, bn=True, relu=False, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.get_2d_file = adaptive_depth_compression(comp_rate_in=0.75, comp_rate_be=0.5, comp_ch=1, kernel_size=1)
        self.fc = nn.Linear(num_features, num_classes)
        self.fc_1 = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # ------ 3D class part ------
        features = self.features(x)
        out = F.relu(features, inplace=True)
        x_out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1)).view(features.size(0), -1)  # 4 1024
        out = self.classifier(x_out)  # 4 2
        # ------ 2D class part ------
        x_2d = self.get_2d_file(x)  # 4 1 64 64 64 -> 4 1 64 64
        x_pfc_2d = self.pfc_2d(x_2d)  # 4 1 64 64 -> 4 64 64 64
        x_pe_0_2d = self.drop_2d(self.pe_0_2d(x_pfc_2d))  # 4 64 64 64 -> 4 64 32 32
        x_pe_1_2d = self.drop_2d(self.pe_1_2d(x_pe_0_2d))  # 4 64 32 32 -> 4 128 16 16
        x_pe_2_2d = self.drop_2d(self.pe_2_2d(x_pe_1_2d))  # 4 128 16 16 -> 4 256 8 8
        x_pe_3_2d = self.drop_2d(self.pe_3_2d(x_pe_2_2d))  # 4 256 8 8 -> 4 512 2 2
        x_out_2d = self.avgpool_2d(x_pe_3_2d)  # 4 512 2 2 -> 4 1024 1 1

        x_out_2d = x_out_2d.view(x_out_2d.size(0), -1)  # 4 1024 1 1 -> 4 1024
        x_out_class_2d = self.fc(x_out_2d)  # 4 512 -> 4 2
        # ------ 3D & 2D output ------
        x_out_total = 0.2 * x_out_2d + 0.8 * x_out
        x_out_class = self.fc_1(x_out_total)  # 4 512 -> 4 2

        return x_out_class, out, x_out_class_2d  # 4 1 64 64 64 -> 4 1024 4 2 2 -> 4 1024 -> 4 2


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)

    return model


if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 64
    x = torch.Tensor(1, 1, image_size, image_size, image_size)
    x = x.to(device)
    print("x size: {}".format(x.size()))
    
    model = generate_model(121,n_input_channels=1,num_classes=2).to(device)
    

    out1 = model(x)
    print("out size: {}".format(out1.size()))
