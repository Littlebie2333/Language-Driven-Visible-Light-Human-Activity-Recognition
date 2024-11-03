import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]  # 去掉末尾多余的时间步


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.chomp = Chomp1d((kernel_size - 1) * dilation)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.leaky_relu1, self.dropout1, self.conv2, self.leaky_relu2,
                                 self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.leaky_relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.leaky_relu(out + res)


class TCNencoder(nn.Module):
    def __init__(self, num_inputs=4, num_channels=[32, 64, 128, 256, 512], kernel_size=3, dropout=0.2):
        super(TCNencoder, self).__init__()
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        # 添加 InstanceNorm1d 层
        self.instance_norm = nn.InstanceNorm1d(num_inputs)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [ResidualBlock(in_channels, out_channels, kernel_size, dilation_size, dropout)]
            layers += [nn.AvgPool1d(kernel_size=3)]  # 添加平均池化层，每层减半序列长度

        self.network = nn.Sequential(*layers)

        # 添加线性层，将特征维度从 512 映射到 768
        self.linear = nn.Linear(512, 768)

    def forward(self, x):
        # x = x[:, :12, :]
        indices = [0, 4, 8, 12]
        x = x[:, indices, :]
        x = self.instance_norm(x)
        x = self.network(x)
        # 交换维度，使其形状变为 (batch_size, sequence_length, 512)
        x = x.permute(0, 2, 1)
        # 使用线性层进行特征维度映射
        x = self.linear(x) #(batch_size, sequence_length, 768)
        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)##标准化

        return x


# # 创建 TCN 模型
# model = TCNencoder()
#
# # 创建输入张量，形状为 (batch_size, num_inputs, sequence_length)
# batch_size = 8
# num_inputs = 16
# sequence_length = 2304
# input_tensor = torch.rand(batch_size, num_inputs, sequence_length)
#
# # 前向传播
# output_tensor = model(input_tensor)
#
# # 打印输出张量的形状
# print(output_tensor.shape)  # 输出形状应为 (batch_size, 768, sequence_length)
