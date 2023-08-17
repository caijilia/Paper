from typing import Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
from timm.models.layers import DropPath, to_2tuple

class PFC(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv3d(in_ch, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels))
        self.depthwise = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, groups=channels, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels))
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(channels))

    def forward(self, x_in):
        x_in = self.input_layer(x_in)
        residual = x_in
        x = self.depthwise(x_in)
        x = x + residual
        x = self.pointwise(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class IRE(nn.Module):
    def __init__(self, in_ch, rate, only_ch=0, only_sp=0):
        super(IRE, self).__init__()
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

        """ # 再加上上下文注意力  或许是上下文吧
        c_in = s  # 8 256 12 16
        c = self.fc3(s)  # 8 256 12 16 -> 8 64 12 16
        c = self.relu(c)
        c = self.fc4(c)  # 8 64 12 16 -> 8 256 12 16
        c = self.sigmoid(c) * c_in  # 8 256 12 16 -> 8 256 12 16 """

        return s


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


class CEE(nn.Module): # Context Enhanced Encoder 3D_version
    def __init__(self, patch_size=3, stride=2, in_chans=64, embed_dim=64, double_branch=1, use_att=0):
        super().__init__()
        self.att_use = use_att
        self.att = IRE(in_ch=embed_dim, rate=4, only_ch=0, only_sp=0)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2)),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        )
        self.proj_c = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dwconv = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True, groups=embed_dim),
            nn.GELU()
        )
        self.fc0 = Conv(embed_dim, embed_dim, 3, bn=True, relu=True)
        self.dwconv_1 = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU()
        )
        self.use_double_branch = double_branch
        self.dwconv_2 = nn.Sequential(
            nn.Conv3d(embed_dim*2, embed_dim, kernel_size=3, padding=1, stride=1, bias=True , groups=embed_dim),
            nn.GELU(),
            nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        self.turn_channel = nn.Sequential(
            nn.Conv3d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, d, h, w = x.shape
        # overlap 编码
        x_pe = self.proj(x) # 在进入第一次编码层前会使用PFC层 # 16 16 32 32 32 -> 16 32 16 16 16
        if self.att_use == 1:
            x_pe = self.att(x_pe)
        # conv 编码
        x_pe_conv = self.proj_c(x) # 16 16 32 32 32 -> 16 32 16 16 16
        # fc_0
        x_PE = x_pe.flatten(2).transpose(1, 2) # 16 32 16 16 16 ->16 32 16*16*16 -> 16 16**3 32
        x_PE = self.norm(x_PE)
        x_po = self.dwconv(x_pe).flatten(2).transpose(1, 2) # 按照 unext 这里是加入位置编码 # 16 32 16 16 16 ->16 32 16*16*16 -> 16 16**3 32
        x_0  = torch.transpose((x_PE + x_po), 1, 2).view(b, x_pe.shape[1], int(d/2), int(h/2), int(w/2)) # 16 16**3 32 -> 16 32 16*16*16 -> 16 32 16 16 16
        x_0  = self.fc0(x_0) # 16 32 16 16 16 
        # fc_1
        x_1  = x_0 
        if self.use_double_branch == 1:
            x_1_ = self.dwconv_2(torch.cat([x_1, x_pe_conv], dim=1))
            x_1_ = self.turn_channel(torch.cat([x_1_, x_pe], dim=1)).flatten(2).transpose(1, 2)
            x_out  = torch.transpose((x_1_ + x_PE), 1, 2).view(b, x_pe.shape[1], int(d/2), int(h/2), int(w/2)) # 16 16**3 32 -> 16 32 16*16*16 -> 16 32 16 16 16
            return x_out
        return x_1


class UNETR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
        #------fine unet part------
        self.pfc = PFC(in_ch=1, channels=16, kernel_size=7)
        self.encoder1 = CEE(patch_size=3, stride=2, in_chans=16, embed_dim=32, double_branch=1, use_att=0)
        self.encoder2 = CEE(patch_size=3, stride=2, in_chans=32, embed_dim=64, double_branch=1, use_att=0)
        self.encoder3 = CEE(patch_size=3, stride=2, in_chans=64, embed_dim=128, double_branch=1, use_att=0)
        self.encoder4 = CEE(patch_size=3, stride=2, in_chans=128, embed_dim=256, double_branch=1, use_att=0)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        #------fine unet encode sub_network------
        ##-----fine unet encoding-----
        hidden_states_out = []              # 用来存储编码过程中各个阶段的编码结果
        x_to_encoder = self.pfc(x_in)       # 1 1 32 32 32  -> 1 16 32 32 32
        x_en1 = self.encoder1(x_to_encoder) # 1 16 32 32 32 -> 1 32 16 16 16
        hidden_states_out.append(x_en1)     # [1 32 16 16 16]
        x_en2 = self.encoder2(x_en1)        # 1 16 32 32 32 -> 1 64 16 16 16
        hidden_states_out.append(x_en2)     # [1 32 16 16 16, 1 64 16 16 16]
        x_en3 = self.encoder3(x_en2)        # 1 64 16 16 16 -> 1 128 8 8 8
        hidden_states_out.append(x_en3)     # [1 32 16 16 16, 1 64 16 16 16, 1 128 8 8 8]
        x_en4 = self.encoder4(x_en3)        # 1 128 8 8 8   -> 1 256 4 4 4
        x = x_en4                           # 最后一层编码结果 也是最深层 空间分辨率最小的编码结果
        ##-----fine unet encoding-----
        #------fine unet encode sub_network------
        enc1 = x_en1 # self.encoder1(x_in)
        # x2 = hidden_states_out[3]
        enc2 = x_en2 # self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        # x3 = hidden_states_out[6]
        enc3 = x_en3 # self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        # x4 = hidden_states_out[9]
        enc4 = x_en4 # self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits

# 备选的模块
class DRA(nn.Module): # Differential Regional Attention
    def __init__(self, in_ch=64, class_num=2, value=0.5, simple=1, large=0):
        super(DRA, self).__init__()
        self.threshold = value
        self.s_mode = simple
        self.l_mode = large
        self.sigmoid = nn.Sigmoid()
        self.add_learn = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=1),
            nn.BatchNorm3d(in_ch),
            nn.GELU()
        )
        self.to_class_1 = nn.Conv3d(in_channels=in_ch, out_channels=class_num, kernel_size=1)
        self.to_class_2 = nn.Conv3d(in_channels=in_ch, out_channels=class_num, kernel_size=1)
        self.region_learn = nn.Sequential(
            nn.Conv3d(in_channels=class_num, out_channels=1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.GELU()
        )
    
    def forward(self, x1, x2):
        b, c, d, h, w = x1.shape
        x1_sig = self.sigmoid(x1)
        x2_sig = self.sigmoid(x2)
        if self.s_mode == 1:
            x1_in       = 1.0*(x1_sig > self.threshold) 
            x2_in       = 1.0*(x2_sig > self.threshold) 
            edge_diff   = torch.abs(x1_in - x2_in)          
            x           = (edge_diff) * x1 + x1 
            x           = self.add_learn(x)               
        elif self.l_mode == 1:
            x1_in       = self.to_class_1(x1_sig)
            x2_in       = self.to_class_2(x2_sig)
            x1_in       = 1.0*(x1_in > self.threshold) 
            x2_in       = 1.0*(x2_in > self.threshold) 
            edge_diff   = torch.abs(x1_in - x2_in)
            x           = self.sigmoid(edge_diff) * x1 + x1
            x           = self.add_learn(x) 
        return x


class MRA(nn.Module): # Multiple Receptive-field Aggregation
    def __init__(self, c1_in_channels=64, c2_in_channels=128, c3_in_channels=256, embedding_dim=256, drop_rate=0.2, classes=2):
        super(MRA, self).__init__()
        # embedding_dim 是超参数 统一多尺度feature map的channel维度
        self.conv_c1 = nn.Conv3d(in_channels=c1_in_channels, out_channels=embedding_dim, kernel_size=1)
        self.down_1  = nn.MaxPool3d(kernel_size=4, stride=4, padding=0, dilation=1)
        self.conv_c2 = nn.Conv3d(in_channels=c2_in_channels, out_channels=embedding_dim, kernel_size=1)
        self.down_2  = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1)
        # 三个不同尺度的数据进行融合
        self.conv_fuse = nn.Sequential(
            nn.Conv3d(in_channels=embedding_dim*3, out_channels=embedding_dim, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm3d(embedding_dim),
            nn.GELU()
        )
        self.drop = nn.Dropout3d(drop_rate)
        self.edge = DRA(in_ch=embedding_dim, class_num=classes)
    
    def forward(self, inputs):
        # inputs 包括三个feature map 分别是 x_b_2, x_b_1, x_b
        c1, c2, c3 = inputs

        c1_ = self.conv_c1(c1) # 16 64 48 64 -> 16 256 48 64
        c1_ = self.down_1(c1_) # 16 256 48 64 -> 16 256 12 16

        c2_ = self.conv_c2(c2) # 16 128 24 32 -> 16 256 24 32
        c2_ = self.down_2(c2_) # 16 256 24 32 -> 16 256 12 16

        c3_ = c3 # 16 256 12 16

        c_fuse = self.conv_fuse(torch.cat([c1_, c2_, c3_], dim=1)) # 8 64 48 64 -> 8 64*4 48 64 -> 8 64 48 64
        x = self.drop(c_fuse) + self.edge(c2_, c1_) + self.edge(c3_, c2_)

        return x
