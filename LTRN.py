# -*- coding: utf-8 -*-
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class MSA(nn.Module):
    def __init__(self, num_vector, num_heads_column, heads_number):
        """
        :param num_vector: 向量的个数
        :param num_heads_column: Wk矩阵的列——dim_vector×num_head_column
        :param heads_number: 多头的个数
        """
        super(MSA, self).__init__()
        # print(num_vector,num_heads_column,heads_number)
        self.num_vector = num_vector
        self.num_heads_column = num_heads_column
        self.heads_number = heads_number
        self.to_q = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_k = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_v = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))  # 权重参数*CORE
        self.proj = nn.Linear(num_heads_column * heads_number, num_vector)
        # 位置编码这里 待修改一下，不用M-MSA的
        self.pos_emb = nn.Sequential(
            nn.Conv2d(num_heads_column * heads_number, num_heads_column * heads_number,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.modules.activation.GELU(),
            nn.Conv2d(num_heads_column * heads_number, num_heads_column * heads_number,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, x_in):
        """
        :param x_in:
                    输入的图像形状 b, h, w, c  则是对c进行注意力
                    输入的图像形状 b, w, c, h  则是对h进行注意力
                    输入的图像形状 b, c, h, w  则是对w进行注意力
        :return out_c:    主干的输出的形状与输入一致
                out_p:    位置编码输出的形状与输入一致
                为了方便后续分别进行卷积操作不进行相加了
        """
        b, h, w, c = x_in.shape
        # print (x_in.shape)
        x = x_in.reshape(b, h * w, c)  # x 形状变为 B,H*W,C
        # print (x.shape)
        # print("to_q前{}".format(x.shape))
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # print(v_inp.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp, v_inp))
        v = v
        # q,k,v: b,heads,hw,c   这里的c不再是x_in.shape的c了  应该就是num_heads_column
        q = q.transpose(-2, -1)  # q,k,v: b,heads,c,hw
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.heads_number * self.num_heads_column)
        out_c = self.proj(x).view(b, h, w, c)  # 没有view之前  x的形状是 b,h*w,num_vector  num_vector:x_in.shape的c

        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return [out_c, out_p]


class ResNetBlock2(nn.Module):  # 不是经典的残差块
    def __init__(self, c_in):
        super(ResNetBlock2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=(1, 1), bias=False),  # padding =0
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_in, c_in // 2, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),  # padding =0
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_in // 2, c_in // 2, kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(c_in // 2, c_in, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + x
        out = F.leaky_relu(out, negative_slope=0.2, inplace=False)
        return out


class RTGB1(nn.Module):
    def __init__(self, num_direction):
        """
        平均池化和最大池化组合
        :param num_direction:   要进行全局平均池化的面的个数
                                输入是 b c h w --> b c 1 1   num_direction=c
                                输入是 b w c h --> b w 1 1   num_direction=w
                                输入是 b h w c --> b h 1 1   num_direction=h
        """
        super(RTGB1, self).__init__()
        self.rtgb_average = nn.AdaptiveAvgPool2d((1, 1))
        self.rtgb_max = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_share = nn.Sequential(
            nn.Conv2d(num_direction, num_direction // 2, (1, 1)),  # 不能padding
            nn.Conv2d(num_direction // 2, num_direction, (1, 1)),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_direction, num_direction // 2, (1, 1)),  # 不能padding
            nn.Conv2d(num_direction // 2, num_direction, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """

        :param x:  输入是 b c h w --> b c 1 1
                   输入是 b w c h --> b w 1 1
                   输入是 b h w w --> b h 1 1
        :return:   b c(w)(h) 1 1
        """
        x_average = self.rtgb_average(x)
        x_max = self.rtgb_max(x)
        x_all = self.conv_share(x_average) + self.conv_share(x_max)
        out = self.conv(x_all)
        return out


class RTGB2(nn.Module):
    def __init__(self, num_direction, width, height):
        """
        depthwise
        :param num_direction:   要进行全局平均池化的面的个数
                                输入是 b c h w --> b c 1 1   num_direction=c
                                输入是 b w c h --> b w 1 1   num_direction=w
                                输入是 b h w c --> b h 1 1   num_direction=h
        :param width            输入图片的第三个维度（相对的宽）
        :param height           输入图片的第四个维度（相对的高）
        """
        super(RTGB2, self).__init__()
        self.rtgb_depthwise = nn.Conv2d(num_direction, num_direction, kernel_size=(width, height), groups=num_direction)
        self.conv = nn.Sequential(
            nn.Conv2d(num_direction, num_direction // 2, (1, 1)),  # 不能padding
            nn.Conv2d(num_direction // 2, num_direction, (1, 1)),
            nn.Sigmoid()
            # nn.LayerNorm()
        )

    def forward(self, x):
        """

        :param x:  输入是 b c h w --> b c 1 1
                   输入是 b w c h --> b w 1 1
                   输入是 b h w c --> b h 1 1
        :return:   b c(w)(h) 1 1
        """
        x_depthwise = self.rtgb_depthwise(x)
        out = self.conv(x_depthwise)
        return out


class RTGB3(nn.Module):
    def __init__(self, num_direction, reduction):
        """
        depthwise
        :param num_direction:   要进行全局平均池化的面的个数
                                输入是 b c h w --> b c 1 1   num_direction=c
                                输入是 b w c h --> b w 1 1   num_direction=w
                                输入是 b h w c --> b h 1 1   num_direction=h
        :param width            输入图片的第三个维度（相对的宽）
        :param height           输入图片的第四个维度（相对的高）
        """
        super(RTGB3, self).__init__()
        self.conv = nn.Conv2d(num_direction, 1, (1, 1), bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(num_direction, num_direction // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(num_direction // reduction, num_direction, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h * w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h * w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)  # y: b c

        return self.fc(y)


class TransformerRTGB(nn.Module):
    def __init__(self, c_in, w_in, h_in, heads, rtgb_method):
        """

        :param heads_number 多头数
        :param  c_in, h_in, w_in:   输入图像的通道b c h w
        rtgb_method: 秩一张量生成方法选择
        """
        super(TransformerRTGB, self).__init__()
        # msa的参数：num_vector: 向量的个数  即对谁做注意力机制
        #           num_heads_column: Wk矩阵的列——dim_vector×num_head_column
        #                                  一般的设置为 num_vector/heads_number
        #           heads_number: 多头的个数
        self.msa_c = nn.Sequential(
            MSA(num_vector=c_in, num_heads_column=c_in // heads, heads_number=heads)

        )
        # print("msa_h:")
        self.msa_h = nn.Sequential(
            MSA(num_vector=h_in, num_heads_column=h_in // heads, heads_number=heads)
        )
        # print("msa_w:")
        self.msa_w = nn.Sequential(
            MSA(num_vector=w_in, num_heads_column=w_in // heads, heads_number=heads)
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(c_in, c_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(w_in, w_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(h_in, h_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )

        self.layernorm_c0 = nn.LayerNorm(c_in)
        self.layernorm_w0 = nn.LayerNorm(w_in)
        self.layernorm_h0 = nn.LayerNorm(h_in)

        self.layernorm_c = nn.LayerNorm(c_in)
        self.layernorm_w = nn.LayerNorm(w_in)
        self.layernorm_h = nn.LayerNorm(h_in)

        """         
        self.norm_c =nn.BatchNorm2d(c_in, eps=1e-05, momentum=0.1,
         affine=True, track_running_stats=True)
        self.norm_w =nn.BatchNorm2d(w_in, eps=1e-05, momentum=0.1,
         affine=True, track_running_stats=True)
        self.norm_h =nn.BatchNorm2d(h_in, eps=1e-05, momentum=0.1,
         affine=True, track_running_stats=True) 
         """
        # rtgb:
        # 输入是 b c h w --> b c 1 1
        # 输入是 b w c h --> b w 1 1
        # 输入是 b h w c --> b h 1 1

        # 平均+最大
        if rtgb_method == "RTGB1":
            self.rtgb_c1 = RTGB1(c_in)
            self.rtgb_h1 = RTGB1(h_in)
            self.rtgb_w1 = RTGB1(w_in)
        elif rtgb_method == "RTGB2":
            # depthwise conv：
            self.rtgb_c1 = RTGB2(c_in, h_in, w_in)
            self.rtgb_h1 = RTGB2(h_in, w_in, c_in)
            self.rtgb_w1 = RTGB2(w_in, c_in, h_in)
        elif rtgb_method == "RTGB3":
            #  借鉴AWAC的
            self.rtgb_c1 = RTGB3(c_in, 2)
            self.rtgb_h1 = RTGB3(h_in, 2)
            self.rtgb_w1 = RTGB3(w_in, 2)
        else:
            print("RTGB选择错误")

    def forward(self, x_in):
        """

        :param x_in:   初始形状是 b c w h
        :return: out 是低秩张量  b c w h
        """
        # x_in : b, c, w, h
        b_in, c_in, w_in, h_in = x_in.shape

        x_c = x_in  # b c w h
        x_w = x_in.permute(0, 2, 1, 3)  # b w c h
        x_h = x_in.permute(0, 3, 2, 1)  # b h w c
        x_c = x_c.permute(0, 3, 2, 1)  # x: b  h w c
        # 所有msa的输出形状与输入一致，一个主干编码，一个位置编码
        x_c = self.layernorm_c0(x_c)
        out_c, out_pc = self.msa_c(x_c)
        # print("msa_c")
        x_h = x_h.permute(0, 2, 3, 1)  # b h w  c -->  b, w, c, h
        # print("x1的形状{}".format(x1.shape))
        # print(x1.shape)
        x_h = self.layernorm_h0(x_h)
        out_h, out_ph = self.msa_h(x_h)
        # print("msa_h")
        x_w = x_w.permute(0, 2, 3, 1)  # b w c h -->  b, c, h ,w
        x_w = self.layernorm_w0(x_w)
        out_w, out_pw = self.msa_w(x_w)
        # print("msa_w")
        out_c = out_c + out_pc
        out_h = out_h + out_ph
        out_w = out_w + out_pw
        out_c = self.layernorm_c(out_c)
        out_w = self.layernorm_w(out_w)
        out_h = self.layernorm_h(out_h)
        out_c = self.conv_c(out_c.permute(0, 3, 1, 2) + x_c.permute(0, 3, 1, 2))  # out:b  h w c  -->b c  h w
        out_h = self.conv_h(out_h.permute(0, 3, 1, 2) + x_h.permute(0, 3, 1, 2))  # out:b, w, c, h  --> b h w c
        out_w = self.conv_w(out_w.permute(0, 3, 1, 2) + x_w.permute(0, 3, 1, 2))  # out : b, c, h ,w --> b w c h
        # out 与对应的 x x1 x2形状相同
        # 此处只做了一次self-attention
        # 如果多次的话 需要 1：out_+out_pc
        #                2：做一个线性变换
        #                3：做一个类似FFN的函数
        # rtgb: 输入b c h w --> 输出b c 1 1
        #       输入b h w c --> b h 1 1
        #       输入b w c h --> b w 1 1
        vector_c = self.rtgb_c1(out_c)  # out:b  h w c  -->b c  h w
        vector_h = self.rtgb_h1(out_h)  # out:b, w, c, h  --> b h w c
        vector_w = self.rtgb_w1(out_w)  # out : b, c, h ,w --> b w c h
        assert torch.isnan(vector_c).sum() == 0, print("////")
        assert torch.isnan(vector_h).sum() == 0, print("////")
        assert torch.isnan(vector_w).sum() == 0, print("////")

        # 至此所有out都是向量了: b,c(w)(h),1,1
        # 加一个一个行、列、通道的上下文信息学习
        matrix3d = torch.empty(b_in, w_in, h_in, requires_grad=True).cuda()

        # print(matrix3d.shape)
        for i in range(0, b_in):
            matrix3d[i] = torch.outer(vector_w.view(b_in, -1)[i],
                                      vector_h.view(b_in, -1)[i])  # Kronecker product得到bachsize个W*H的矩阵
        # print("应该是带batch的三维的张量的{}".format(matrix3d.shape))
        mat_3d = matrix3d.detach().cpu().numpy()
        for j in range(0, b_in):
            assert torch.isnan(matrix3d[j]).sum() == 0, print("////")
        assert torch.isnan(matrix3d).sum() == 0, print("////")

        matrix4d = torch.empty(b_in, c_in, w_in, h_in, requires_grad=True).cuda()
        for i in range(0, b_in):
            # print("一个c_1:{}".format(c_1.view(cb,-1)[i].shape))

            # print("一个matrix3d[i]:{}:".format(matrix3d[i].view(-1).shape))
            matrix4d[i] = torch.outer(vector_c.view(b_in, -1)[i], matrix3d[i].view(-1)) \
                .view(c_in, w_in, h_in)  # Kronecker product积操作，得到product得到bachsize个C*(W*H)的二维张量
        assert torch.isnan(matrix4d).sum() == 0, print("////")

        out = matrix4d
        return out


class TransformerDRTLM(nn.Module):
    def __init__(self, c_in, w_in, h_in, heads_num, rtgb_method, rtgb_num):
        """

        :param c_in:    做CP分解的原特征形状   就是cp_in_c
        :param h_in:
        :param w_in:
        :param rtgb_method:
        """
        super(TransformerDRTLM, self).__init__()
        # self.rtgb_layers = nn.ModuleList([])
        # for i in range(rtgb_num):
        #     self.rtgb_layers.append(nn.ModuleList([
        #         TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method),
        #     ]))
        # TransformerRTGB的参数：
        #  c_in, h_in, w_in
        self.rtgb1 = TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method)
        self.rtgb2 = TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method)
        self.rtgb3 = TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method)
        #self.rtgb4 = TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in * rtgb_num, c_in, (3, 3), padding=1),
        )
        # 因为要进行Hadamard product，所示与输入时的通道一致

    def forward(self, x):
        """
        :param x:   输入的形状是 b c w h
        :return: output 形状是  b c w h
        """
        # print("rtgb1{}".format(x.shape))
        # o = []
        # fea = x
        # o_all = []
        # for i, rtgb in enumerate(self.rtgb_layers):
        #     print(i)
        #     if i != 0:
        #         fea = fea - o[i - 1]
        #     fea = rtgb(x)
        #     o.append(fea)
        #     o_all = torch.cat((o_all, fea), dim=1)

        o1 = self.rtgb1(x)  # x是B*C*W*H的  o1也是    o2也是
        o2 = self.rtgb2(x - o1)
        o3 = self.rtgb3(x - o1 - o2)
        #o4 = self.rtgb4(x - o1 - o2 - o3)
        o_all = torch.cat((o1, o2,o3), dim=1)
        #assert torch.isnan(o1).sum() == 0, print("////")
        #assert torch.isnan(o2).sum() == 0, print("////")
        #assert torch.isnan(o3).sum() == 0, print("////")

        # o1 = self.rtgb1(x)  # x是B*C*W*H的  o1也是    o2也是
        # o2 = self.rtgb2(x - o1)
        # o3 = self.rtgb3(x - o1 - o2)
        # o_all = torch.cat((o1, o2, o3), dim=1)

        output = self.conv(o_all)  # 学习CP分解中的各低秩张量系数
        return output  # 已经是CP分解后的秩一张量了


class Transformer_CP(nn.Module):
    def __init__(self, w_in, h_in, cp_in_c, heads_num, rtgb_method, rtgb_num):
        """

        :param w_in:  输入图像的宽
        :param h_in:  输入图像的高
        :param cp_in_c:   做cp分解的时三维数据的通道数，即TransformerRTGB 和 DRTLM 的参数c_in
                          我觉得cp_in_c 应该大于等于 out_c   会好一点？
        :param out_c:      最终超分结果的通道数
        :param       rtgb_method
        """
        super(Transformer_CP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1,
            # affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.transformer_DRTLM = TransformerDRTLM(cp_in_c, w_in, h_in, heads_num, rtgb_method, rtgb_num)
        self.resnet = ResNetBlock2(cp_in_c)

    def forward(self, x):
        """

        :param x:   输入的图像  b c w h
        :return:    形状是  b c w h   # c 应该时cp_in_c
        """

        # print(x_input.shape)

        x1 = self.conv(x)
        assert torch.isnan(x1).sum() == 0, print("////")

        x2 = self.transformer_DRTLM(x1)  # 至此cp分解完成，得到了低秩张量x2  后续进行恢复
        assert torch.isnan(x2).sum() == 0, print("////")

        # x2 : b cp_in_c w h
        out = x2 * x1
        assert torch.isnan(out).sum() == 0, print("////")
        out = self.resnet(out)
        return out


# 注意输入transformer_cp的图篇形状大小问题
#  因为涉及到整除  过程中下采样为1/2 ，n次下采样， num个注意力头  则需要 图片的 h w c 分别别为 2*3*num(4)的整数倍
class UDoubleTransformerCP3(nn.Module):
    def __init__(self, c_in, w_in, h_in, cp_in_c, c_out, heads_num, stage, rtgb_method, rtgb_num):
        super(UDoubleTransformerCP3, self).__init__()
        self.stage = stage
        self.conv0 = nn.Conv2d(c_in, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv00 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_de = nn.Conv2d(cp_in_c, cp_in_c, 4, 2, 1, bias=False)  # 空间上减少一半
        # 前面降维空间上减少了一半，所以除以2
        self.cp1 = Transformer_CP(w_in // 2, h_in // 2, cp_in_c, heads_num, rtgb_method, rtgb_num)
        self.bottleneck1 = nn.Sequential(
            # 下采样操作，通道增加一倍
            nn.Conv2d(cp_in_c, cp_in_c, (4, 4), (2, 2), 1, bias=False),
            nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            # 空间上采用，通道还原
            nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

        # 这里有拼接操作所示 输入的cp_in_c 不用除以2
        self.fusion1 = nn.Conv2d(cp_in_c*2, cp_in_c, (1, 1), (1, 1), bias=False)
        self.cp2 = Transformer_CP(w_in // 2, h_in // 2, cp_in_c, heads_num, rtgb_method, rtgb_num)

        self.conv_up1 = nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.conv_up2 = nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(cp_in_c, 32, (3, 3), (1, 1), 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1, bias=False),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cp_in_c, 32, (3, 3), (1, 1), 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1, bias=False),
            nn.Conv2d(32, c_out, (3, 3), (1, 1), 1, bias=False),
        )
        self.conv1_fu = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)
        self.conv1_fu2 = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)
        self.conv1_fu3 = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)

    def forward(self, x):
        """

        :param x: 输入的图像 b c w h
        :return:
        """
        x0 = self.conv0(x)  # 通道上采用
        x_de = self.conv_de(x0)  # 空间降维
        fea1 = self.cp1(x_de)  # 第一个CP分解块，后续需要使用该fea的
        fea = self.bottleneck1(fea1)  # 包括了下采样-卷积-上采样
        fea = self.fusion1(torch.cat([fea, fea1], dim=1))  # cp1的输出和bottleneck1的输出拼接
        fea2 = self.cp2(fea)
        fea = self.conv1_fu3(fea2)   #cp2的输出1×1卷积一次
        fea = self.conv_up1(fea)      # cp2的输出反卷积空间上采样
        temp_x1 = self.conv1_fu2(fea1)  # cp1的输出1×1卷积一次
        temp_x1 = self.conv_up2(temp_x1) # cp1的输出上采样
        out = self.conv1(fea + temp_x1)   # cp1的输出和cp2输出相加后进行重建卷积
        temp_x0 = self.conv1_fu(x0)    # 浅层次特征 空间降维前的
        out = self.conv2(temp_x0 + out)
        out1 = self.conv00(x)     # 直接从原图输出31的跳跃连接
        out = out1 + out
        return out

#
