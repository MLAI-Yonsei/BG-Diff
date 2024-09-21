import torch
from torch import nn, einsum
import torch.nn.functional as F
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import default, LinearAttention, Residual, PreNorm, Downsample, Upsample, partial, Attention, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, ResnetBlock
from denoising_diffusion_pytorch.nn import normalization, AttentionPool2d,zero_module, conv_nd
from denoising_diffusion_pytorch.resnet import MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
from utils import Lambda

class Classifier(nn.Module):
    def __init__(self, image_size, num_classes, t_dim=1) -> None:
        super().__init__()
        self.linear_t = nn.Linear(t_dim, num_classes)
        self.linear_img = nn.Linear(image_size * image_size * 3, num_classes)
    def forward(self, x, t):
        """
        Args:
            x (_type_): [B, 3, N, N]
            t (_type_): [B,]

        Returns:
                logits [B, num_classes]
        """
        B = x.shape[0]
        t = t.view(B, 1)
        logits = self.linear_t(t.float()) + self.linear_img(x.view(x.shape[0], -1))
        return logits
    
class Regressor(nn.Module):
    def __init__(self, seq_len, num_classes=2, t_dim=1) -> None:
        super().__init__()
        self.linear_t = nn.Linear(t_dim, num_classes) 
        self.linear_seq = nn.Linear(seq_len, num_classes)  

    def forward(self, x, t):
        """
        Args:
            x (_type_): [B, 1, Seq_len]
            t (_type_): [B,]

        Returns:
                outputs [B, num_classes]
        """
        B = x.shape[0]
        t = t.view(B, 1)
        outputs = self.linear_t(t.float()) + self.linear_seq(x.view(x.shape[0], -1))
        return outputs
    
class MLPRegressor(nn.Module):
    def __init__(self, ch, dims, out_channels=2, num_head_channels=-1, pool="adaptive"):
        super().__init__()
        if pool == "adaptive":
            self.out = nn.Sequential(
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
                zero_module(conv_nd(1, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                nn.SiLU(),
                AttentionPool2d(
                    (4), ch, num_head_channels, out_channels
                    # (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")
        
    def forward(self, x):
        return self.out(x)

class Unet1DEncoder(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 5,#8
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        pool = "adaptive"
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim - 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        num_groups=6
        self.group_emb = nn.Embedding(num_groups, time_dim)
        self.out_mlp = MLPRegressor(dims[-1], dims[-1], 2, pool=pool)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim * 2),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2) # time_emb_dim = time_dim
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)

    def forward(self, x, time, group, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        x = self.init_conv(x)
        t = self.time_mlp(time)
        g = self.group_emb(group)
        tg = torch.cat((t, g), dim=-1)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, tg)
            x = block2(x, tg)
            x = attn(x)
            x = downsample(x)

        x = self.mid_block1(x, tg)
        x = self.mid_attn(x)
        emb = self.mid_block2(x, tg)

        return self.out_mlp(emb), emb

class ddResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """
    def __init__(self, in_channels=1, base_filters=32, first_kernel_size=5, kernel_size=3, stride=4, 
                        groups=2, n_block=8, output_size=2 , is_se=False, se_ch_low=4, downsample_gap=2, 
                        increasefilter_gap=2, use_bn=True, use_do=True, self_condition=False, final_layers=1, concat_label_mlp=False, g_pos="rear", g_mlp_layers=3):
        super(ddResNet1D, self).__init__()
        
        self.n_block = n_block
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.is_se = is_se
        self.se_ch_low = se_ch_low
        self.channels = in_channels
        self.self_condition = self_condition
        self.concat_label_mlp = concat_label_mlp
        self.g_pos = g_pos
        
        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.first_kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block,
                is_se=self.is_se,
                se_ch_low=self.se_ch_low)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

        # condtional layer
        fourier_dim = 256; time_dim = 625
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim), # fourier_dim - 1 ?
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            Lambda(lambda x: x.unsqueeze(1)) # Add a dimension at position 1
        )
        # num_groups=6
        # self.group_emb = nn.Embedding(num_groups, time_dim)
        if self.g_pos=="front":
            g_mlp_dim=625
        elif self.g_pos=="rear":
            g_mlp_dim=256
        if not self.concat_label_mlp :
            self.sp_encoder = MLP(input_dim = 1, hidden_dim = g_mlp_dim, output_dim = g_mlp_dim, num_layers = g_mlp_layers)
            self.dp_encoder = MLP(input_dim = 1, hidden_dim = g_mlp_dim, output_dim = g_mlp_dim, num_layers = g_mlp_layers)
        else:
            self.spdp_encoder = MLP(input_dim = 2, hidden_dim = g_mlp_dim, output_dim = g_mlp_dim, num_layers = g_mlp_layers)
        
        # Classifier
        # self.main_clf = nn.Linear(out_channels, output_size)
        self.main_clf = MLP(input_dim = out_channels, hidden_dim = out_channels, output_dim = output_size, num_layers = final_layers)

    def forward(self, x, t=None, g=None):
        # x = x['ppg']
        assert len(x.shape) == 3
        assert g != None
        assert t != None
        assert x.shape[0] == t.shape[0]

        t = self.time_layer(t)
        assert x.shape == t.shape
        # Condition with label g and t
        if self.g_pos == "front":
            if not self.concat_label_mlp :
                t = t + self.sp_encoder(g[:,0].unsqueeze(1).type(torch.float32)).unsqueeze(1) + self.sp_encoder(g[:,1].unsqueeze(1).type(torch.float32)).unsqueeze(1)    
            else:
                t = t + self.spdp_encoder(g.type(torch.float32)).unsqueeze(1)
        x = x + t 

        # skip batch norm if batchsize<4:
        if x.shape[0]<4:    self.use_bn = False 
        # first conv
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)
        
        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        h = self.final_relu(out)
        h = h.mean(-1) # (n_batch, out_channels)
        # logger.info('final pooling', h.shape)

        # Condition with label g
        if self.g_pos == "rear":
            if not self.concat_label_mlp :
                h = h + self.sp_encoder(g[:,0].unsqueeze(1).type(torch.float32)) + self.sp_encoder(g[:,1].unsqueeze(1).type(torch.float32))    
            else:
                h = h + self.spdp_encoder(g.type(torch.float32))

        # ===== Concat x_demo
        out = self.main_clf(h)
        return out
    
class no_lightening_ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """
    def __init__(self, in_channels=1, base_filters=32, first_kernel_size=5, kernel_size=3, stride=4, 
                        groups=2, n_block=8, output_size=2 , is_se=False, se_ch_low=4, downsample_gap=2, 
                        increasefilter_gap=2, use_bn=True, use_do=True, self_condition=False, final_layers=1, concat_label_mlp=False, g_pos="rear", g_mlp_layers=3):
        super(no_lightening_ResNet1D, self).__init__()
        
        self.n_block = n_block
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.is_se = is_se
        self.se_ch_low = se_ch_low
        self.channels = in_channels
        self.self_condition = self_condition
        
        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.first_kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block,
                is_se=self.is_se,
                se_ch_low=self.se_ch_low)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        
        # Classifier
        self.main_clf = nn.Linear(out_channels, output_size)
        # self.main_clf = MLP(input_dim = out_channels, hidden_dim = out_channels, output_dim = output_size, num_layers = final_layers)

    def forward(self, x, t=None, g=None):
        # x = x['ppg']

        # skip batch norm if batchsize<4:
        if x.shape[0]<4:    self.use_bn = False 
        # first conv
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)
        
        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        h = self.final_relu(out)
        h = h.mean(-1) # (n_batch, out_channels)
        # logger.info('final pooling', h.shape)

        # ===== Concat x_demo
        out = self.main_clf(h)
        return out
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16, se_ch_low=4):
        super().__init__()
        h = c // r
        if h<4: h = se_ch_low
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, h, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(h, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.414)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        if num_layers == 1:
            # If num_layers is 1, create a Linear layer
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Create the input layer
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Create hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Create the output layer
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)  # No activation for the last layer
            else:
                x = F.relu(layer(x))
        return x
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, first_kernel_size, kernel_size, stride, 
                        groups, n_block, output_size, is_se=False, se_ch_low=4, downsample_gap=2, 
                        increasefilter_gap=2, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.is_se = is_se
        self.se_ch_low = se_ch_low

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.first_kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
        out_channels = base_filters
        # print(f"out channels: {out_channels}")
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            # print(f"out channels: {out_channels}")
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block,
                is_se=self.is_se,
                se_ch_low=self.se_ch_low)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # print(f"out channels: {out_channels}")
        # Classifier
        self.main_clf = nn.Linear(out_channels, output_size)
        
    # def forward(self, x):
    def forward(self, x):
        # x = x['ppg']
        if len(x.shape) != 3:
            assert len(x.shape) == 3

        # skip batch norm if batchsize<4:
        if x.shape[0]<4:    self.use_bn = False 
        # first conv
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)
        # print("out.shape: ", out.shape)
        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        h = self.final_relu(out)
        # print("h.shape: ", h.shape)
        h = h.mean(-1) # (n_batch, out_channels)
        # print("h.shape: ", h.shape)
        # logger.info('final pooling', h.shape)
        # ===== Concat x_demo
        out = self.main_clf(h)
        return out, h
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16, se_ch_low=4):
        super().__init__()
        h = c // r
        if h<4: h = se_ch_low
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, h, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(h, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)