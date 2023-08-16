# 利用这个脚本尝试将我的模型转成 onnx 模型
# 甚至有可能的话 测试一下实际部署的性能
import torch
import torch.nn as nn
# from timm.models.layers import DropPath, to_2tuple


print(torch.__version__)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.GELU()
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
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
    
    
class IRE(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0):
        super(IRE, self).__init__()
        self.fc1        = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/rate), kernel_size=1)
        self.relu       = nn.ReLU(inplace=True)
        self.fc2        = nn.Conv2d(in_channels=int(in_ch/rate), out_channels=in_ch, kernel_size=1)
        self.sigmoid    = nn.Sigmoid()

        self.compress   = ChannelPool()
        self.spatial    = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        self.fc3        = nn.Conv2d(in_channels=in_ch, out_channels=int(in_ch/rate), kernel_size=1)
        self.fc4        = nn.Conv2d(in_channels=int(in_ch/rate), out_channels=in_ch, kernel_size=1)

        self.ch_use     = only_ch 
    
    def forward(self, x):
        x_in = x                
        x = x.mean((2, 3), keepdim=True) 
        x = self.fc1(x)         
        x = self.relu(x)        
        x = self.fc2(x)         
        x = self.sigmoid(x) * x_in 
        if self.ch_use == 1:
            return x
        elif self.ch_use == 0:
            x = x

        s_in = x                    
        s = self.compress(x)        
        s = self.spatial(s)         
        s = self.sigmoid(s) * s_in  

        c_in = s                    
        c = self.fc3(s)             
        c = self.relu(c)
        c = self.fc4(c)             
        c = self.sigmoid(c) * c_in  
    
        return c


class CEE(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_chans=64, embed_dim=64, smaller=0, use_att=0):
        super().__init__()
        self.att_use = use_att
        self.att = IRE(in_ch=embed_dim, rate=4, only_ch=0)
        # patch_size = to_2tuple(patch_size)
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=(3 // 2)),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.proj_c = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True, groups=embed_dim),
            nn.GELU()
        )
        self.fc0 = Conv(embed_dim, embed_dim, 3, bn=True, relu=True)
        self.dwconv_1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU()
        )
        self.use_small_conv = smaller
        self.dwconv_2 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.turn_channel = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_pe = self.proj(x)
        if self.att_use == 1:
            x_pe = self.att(x_pe)
        x_pe_conv = self.proj_c(x)
        x_PE = x_pe.flatten(2).transpose(1, 2) 
        x_PE = self.norm(x_PE)
        x_po = self.dwconv(x_pe).flatten(2).transpose(1, 2) 
        x_0  = torch.transpose((x_PE + x_po), 1, 2).view(b, x_pe.shape[1], int(h/2), int(w/2))
        x_0  = self.fc0(x_0) 
        x_1  = x_0 
        if self.use_small_conv == 1:
            x_1_ = self.dwconv_2(torch.cat([x_1, x_pe_conv], dim=1))
            x_1_ = self.turn_channel(torch.cat([x_1_, x_pe], dim=1)) # .flatten(2).transpose(1, 2)
            # x_out  = x_1_ + x_PE
            x_out = x_1_
            return x_out
        else:
            x_1_ = self.dwconv_1(x_1) 
            x_1_ = self.turn_channel(torch.cat([x_1, x_pe], dim=1)) # .flatten(2).transpose(1, 2)
            # x_out  = self.fc1(x_1_) + x_PE
            x_out = x_1_
            return x_out


class model(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv = nn.Conv2d(3,3,3)
    
    def forward(self, x):
        for i in range(self.n):
            x = self.conv(x)
        return x


model = CEE(smaller=0)
dummy_in = torch.rand(1,64,192,256)
dummy_out = model(dummy_in)
model_trace = torch.jit.trace(model, dummy_in)

torch.onnx.export(model_trace, dummy_in, '/Data2/lh03/files/learngit/models/cee_model.onnx')


print("done")

