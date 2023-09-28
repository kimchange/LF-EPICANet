import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
import math


class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        channels = 64
        self.channels = channels
        self.angRes = angRes
        self.factor = factor
        layer_num = 6


        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )

        self.conv_init = nn.Sequential(
            RES2BLOCK(channels),
            RES2BLOCK(channels),
            RES2BLOCK(channels),
            #   nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            #   nn.LeakyReLU(0.2, inplace=True),
            # #   CascadedBlocks(layer_num, channels, angRes),
            #   nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            #   nn.LeakyReLU(0.2, inplace=True),
            #   nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),

          )


        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = self.make_layer(layer_num=layer_num)

        self.convalt = nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            # nn.Conv3d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv3d(channels, 1, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
        )

    def make_layer(self, layer_num):
        layers = []
        # layers.append(C42_Trans_serial(self.angRes, self.channels, self.MHSA_params, layer_num))
        for i in range(layer_num):
            layers.append(EPITRANS_SANDGLASS(self.angRes, self.channels, self.MHSA_params))
            # layers.append(CascadedBlocks(1, self.channels, self.angRes))
            # layers.append(C42_Trans_parallel(self.angRes, self.channels, self.MHSA_params))
        # layers.append(nn.Conv3d(self.channels, self.channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, lr):
        # Bicubic

        b,u,v,n,h,w = lr.shape[0], self.angRes, self.angRes, self.angRes*self.angRes,lr.shape[2]//self.angRes, lr.shape[3]//self.angRes
        H,W = h*self.factor, w*self.factor
        lr = lr.reshape(b, lr.shape[1], u, h, v, w)
        lr = lr.permute(0,1,2,4,3,5)
        lr = lr.reshape(b, lr.shape[1], n, h, w) #b,1,n,h,w

        # Bicubic
        lr_upscale = F.interpolate(lr.reshape(b* lr.shape[1]* n, 1, h, w),scale_factor= self.factor, mode='bicubic', align_corners=False).reshape(b,1,n,H,W)



        # Initial Convolution
        buffer_init = self.conv_init0(lr)
        # buffer = self.conv_init(buffer_init)+buffer_init
        # buffer = F.leaky_relu(buffer, negative_slope=0.2, inplace=True)
        buffer = self.conv_init(buffer_init)


        # EPIXTrans
        buffer = self.convalt(self.altblock(buffer)) + buffer


        buffer = buffer.permute(0, 2, 1, 3, 4).reshape(b*n, self.channels, h, w)

        buffer = self.upsampling(buffer).reshape(b,n, 1, H, W).permute(0, 2, 1, 3, 4)
        out = buffer + lr_upscale
        

        out = out.reshape(b,1,u,v,H,W).permute(0,1,2,4,3,5).reshape(b,1,u*H,v*W)
        return out


class RES2BLOCK(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Sequential(*[
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        ])
    def forward(self, x):
        y = self.conv(x)
        y = y + x
        return self.act(y)


class EpiXTrans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(EpiXTrans, self).__init__()
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, 
                                               MHSA_params['num_heads'], 
                                               MHSA_params['dropout'], 
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim*2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels, bias=False)

    def gen_mask(self, h: int, w: int, maxdisp: int=2):
        attn_mask = torch.zeros([h, w, h, w])
        # k_h_left = k_h // 2
        # k_h_right = k_h - k_h_left
        # k_w_left = k_w // 2
        # k_w_right = k_w - k_w_left
        [ii,jj] = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')

        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[(ii-i).abs() * maxdisp >= (jj-j).abs()] = 1
                # temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        # attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.reshape(h*w, h*w)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)
        attn_mask = self.gen_mask(v, w, ).to(buffer.device)

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = buffer.permute(3,5,0,2,4,1).reshape(v*w, b*u*h, c)
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = epi_token.reshape(v,w, b,u,h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer

class AngTrans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(AngTrans, self).__init__()
        # self.angRes = angRes
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, self.emb_dim, bias=False)
        self.norm = nn.LayerNorm(self.emb_dim)
        self.attention = nn.MultiheadAttention(self.emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, self.emb_dim * 4, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.emb_dim * 4, self.emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    def forward(self, buffer):
        b,c,u,v,h,w = buffer.shape
        ang_token = buffer.permute(2,3,0,4,5,1).reshape(u*v, b*h*w, c)

        ang_token = self.linear_in(ang_token)

        # ang_PE = self.linear_in(self.ang_position.squeeze(4).squeeze(3).permute(2, 0, 1)) # u*v, 1, c
        # ang_token_norm = self.norm(ang_token + ang_PE)

        # ang_PE = self.linear_in(self.ang_position.squeeze(4).squeeze(3).permute(2, 0, 1)) # u*v, 1, c
        ang_token_norm = self.norm(ang_token )


        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        # buffer = self.Token2SAI(ang_token)
        buffer = ang_token.reshape(u,v, b,h,w, c).permute(2,5,0,1,3,4).reshape(b, c, u, v, h, w)
        return buffer

class EpiTransSandglass(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(EpiTransSandglass, self).__init__()
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        # self.epi_shift_dist = epi_shift_dist
        self.num_heads = MHSA_params['num_heads']
        self.head_dim = emb_dim // self.num_heads
        self.attention = nn.MultiheadAttention(emb_dim, 
                                               MHSA_params['num_heads'], 
                                               MHSA_params['dropout'], 
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim*2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels, bias=False)

    # the ablation of the work can be done by modifying this function
    # when LF SSRx4 task, maskdisp=1
    
    def gen_mask(self, h: int, w: int, maxdisp: int=2):
        attn_mask = torch.zeros([h, w, h, w])

        [ii,jj] = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')

        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[(ii-i).abs() * maxdisp +0 >= (jj-j).abs()] = 1
                # temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        # attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.reshape(h*w, h*w)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

    # def gen_mask(self, h: int, w: int, k_h: int=10, k_w: int=11):
    #     attn_mask = torch.zeros([h, w, h, w])
    #     k_h_left = k_h // 2
    #     k_h_right = k_h - k_h_left
    #     k_w_left = k_w // 2
    #     k_w_right = k_w - k_w_left
    #     for i in range(h):
    #         for j in range(w):
    #             temp = torch.zeros(h, w)
    #             temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
    #             attn_mask[i, j, :, :] = temp

    #     # attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
    #     attn_mask = attn_mask.reshape(h*w, h*w)
    #     attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

    #     return attn_mask

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)
        attn_mask = self.gen_mask(v, w, ).to(buffer.device)

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = buffer.permute(3,5,0,2,4,1).reshape(v*w, b*u*h, c)
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)


        epi_token = self.attention(query=epi_token_norm,
                                    key=epi_token_norm,
                                    value=epi_token,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0] + epi_token


        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = epi_token.reshape(v,w, b,u,h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer




class EPITRANS_SANDGLASS(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(EPITRANS_SANDGLASS, self).__init__()
        self.angRes = angRes

        self.shift_epi_trans = EpiTransSandglass(channels, channels*2, MHSA_params)
        # self.shift_epi_trans = EpiTransSandglass(channels, channels, MHSA_params)
        # self.ang_trans = AngTrans(channels, channels, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

        # self.conv_2 = nn.Sequential(
        #     nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        # )
    def forward(self, x):
        
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()
        
        u, v = self.angRes, self.angRes


        shortcut = x

        # EPI uh
        buffer = x.reshape(b,c,u,v,h,w).permute(0,1,3,2,5,4)
        buffer = self.conv_1( self.shift_epi_trans(buffer).permute(0,1,3,2,5,4).reshape(b,c,n,h,w) ) + shortcut
        # shortcut = buffer

        # EPI vw
        buffer = buffer.reshape(b,c,u,v,h,w)
        buffer = self.conv_1( self.shift_epi_trans(buffer).reshape(b,c,n,h,w) ) + shortcut
        # shortcut = buffer

        # # Ang uv
        # buffer = buffer.reshape(b,c,u,v,h,w)
        # buffer = self.conv_1( self.ang_trans(buffer).reshape(b,c,n,h,w) ) + shortcut
        # # shortcut = buffer

        return buffer


class SpatialConv(nn.Module):
    def __init__(self, ch):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1)),
                    nn.LeakyReLU(0.2, inplace=True)
                    )

    def forward(self,fm):

        return self.spaconv_s(fm) 



class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR):
        loss = self.criterion_Loss(SR, HR)

        return loss


if __name__ == "__main__":
    net = Net(5, 4).cuda()
    from thop import profile
    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
