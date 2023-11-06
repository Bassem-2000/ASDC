import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, \
    AdaptiveAvgPool2d, Sequential, Module

class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                           stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x
    
class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                           groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class Depth_Wise(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True,
                                      kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)
    
    
class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x
    
class ResidualSE(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block-1:
                modules.append(
                    Depth_Wise_SE(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride,
                               groups=groups, se_reduct=se_reduct))
            else:
                modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding,
                                          stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Depth_Wise_SE(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=8):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output
    
class anti_spoofing(Module):
    def __init__(self, embedding_size=128, conv6_kernel=(5, 5), drop_p=0.2, num_classes=3, img_channel=3):
        super(anti_spoofing, self).__init__()
        keep_dict = {'1.8M_': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
                               13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
                               52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
                               128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]}

        self.embedding_size = embedding_size

        self.conv1 = Conv_block(img_channel, keep_dict['1.8M_'][0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep_dict['1.8M_'][0], keep_dict['1.8M_'][1], kernel=(3, 3), stride=(1, 1),
                                   padding=(1, 1), groups=keep_dict['1.8M_'][1])

        c1 = [(keep_dict['1.8M_'][1], keep_dict['1.8M_'][2])]
        c2 = [(keep_dict['1.8M_'][2], keep_dict['1.8M_'][3])]
        c3 = [(keep_dict['1.8M_'][3], keep_dict['1.8M_'][4])]

        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                  groups=keep_dict['1.8M_'][4])

        c1 = [(keep_dict['1.8M_'][4], keep_dict['1.8M_'][5]), (keep_dict['1.8M_'][7], keep_dict['1.8M_'][8]),
              (keep_dict['1.8M_'][10], keep_dict['1.8M_'][11]), (keep_dict['1.8M_'][13], keep_dict['1.8M_'][14])]
        c2 = [(keep_dict['1.8M_'][5], keep_dict['1.8M_'][6]), (keep_dict['1.8M_'][8], keep_dict['1.8M_'][9]),
              (keep_dict['1.8M_'][11], keep_dict['1.8M_'][12]), (keep_dict['1.8M_'][14], keep_dict['1.8M_'][15])]
        c3 = [(keep_dict['1.8M_'][6], keep_dict['1.8M_'][7]), (keep_dict['1.8M_'][9], keep_dict['1.8M_'][10]),
              (keep_dict['1.8M_'][12], keep_dict['1.8M_'][13]), (keep_dict['1.8M_'][15], keep_dict['1.8M_'][16])]

        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep_dict['1.8M_'][4], kernel=(3, 3),
                               stride=(1, 1), padding=(1, 1))

        c1 = [(keep_dict['1.8M_'][16], keep_dict['1.8M_'][17])]
        c2 = [(keep_dict['1.8M_'][17], keep_dict['1.8M_'][18])]
        c3 = [(keep_dict['1.8M_'][18], keep_dict['1.8M_'][19])]

        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                  groups=keep_dict['1.8M_'][19])

        c1 = [(keep_dict['1.8M_'][19], keep_dict['1.8M_'][20]), (keep_dict['1.8M_'][22], keep_dict['1.8M_'][23]),
              (keep_dict['1.8M_'][25], keep_dict['1.8M_'][26]), (keep_dict['1.8M_'][28], keep_dict['1.8M_'][29]),
              (keep_dict['1.8M_'][31], keep_dict['1.8M_'][32]), (keep_dict['1.8M_'][34], keep_dict['1.8M_'][35])]
        c2 = [(keep_dict['1.8M_'][20], keep_dict['1.8M_'][21]), (keep_dict['1.8M_'][23], keep_dict['1.8M_'][24]),
              (keep_dict['1.8M_'][26], keep_dict['1.8M_'][27]), (keep_dict['1.8M_'][29], keep_dict['1.8M_'][30]),
              (keep_dict['1.8M_'][32], keep_dict['1.8M_'][33]), (keep_dict['1.8M_'][35], keep_dict['1.8M_'][36])]
        c3 = [(keep_dict['1.8M_'][21], keep_dict['1.8M_'][22]), (keep_dict['1.8M_'][24], keep_dict['1.8M_'][25]),
              (keep_dict['1.8M_'][27], keep_dict['1.8M_'][28]), (keep_dict['1.8M_'][30], keep_dict['1.8M_'][31]),
              (keep_dict['1.8M_'][33], keep_dict['1.8M_'][34]), (keep_dict['1.8M_'][36], keep_dict['1.8M_'][37])]

        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep_dict['1.8M_'][19], kernel=(3, 3),
                               stride=(1, 1), padding=(1, 1))

        c1 = [(keep_dict['1.8M_'][37], keep_dict['1.8M_'][38])]
        c2 = [(keep_dict['1.8M_'][38], keep_dict['1.8M_'][39])]
        c3 = [(keep_dict['1.8M_'][39], keep_dict['1.8M_'][40])]

        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                  groups=keep_dict['1.8M_'][40])

        c1 = [(keep_dict['1.8M_'][40], keep_dict['1.8M_'][41]), (keep_dict['1.8M_'][43], keep_dict['1.8M_'][44])]
        c2 = [(keep_dict['1.8M_'][41], keep_dict['1.8M_'][42]), (keep_dict['1.8M_'][44], keep_dict['1.8M_'][45])]
        c3 = [(keep_dict['1.8M_'][42], keep_dict['1.8M_'][43]), (keep_dict['1.8M_'][45], keep_dict['1.8M_'][46])]

        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep_dict['1.8M_'][40], kernel=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(keep_dict['1.8M_'][46], keep_dict['1.8M_'][47], kernel=(1, 1), stride=(1, 1),
                                      padding=(0, 0))
        self.conv_6_dw = Linear_block(keep_dict['1.8M_'][47], keep_dict['1.8M_'][48], groups=keep_dict['1.8M_'][48],
                                      kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        print(out.shape)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out