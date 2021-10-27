import torch
import torch.nn as nn    # 神经网络的层
import torch.utils.data as Data


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 卷积层
            nn.Conv2d(      # (1, 64, 64)
                in_channels=1,  # 输入的通道数
                out_channels=16,  # 卷积核的高度
                kernel_size=7,  # 卷积核的大小
                stride=1,  # 步长
                padding="same",  # 边缘填充，if stride=1,padding="same",卷积后图片大小不变
            ),
            nn.BatchNorm2d(16),  # 批归一化
            nn.ReLU(),  # 激励函数
        )

        self.conv2 = nn.Sequential(  # 第二个卷积
            nn.Conv2d(16, 32, 5, 1, "same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(  # 第三个卷积
            nn.Conv2d(32, 32, 5, 1, "same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(  # 第四个卷积
            nn.Conv2d(32, 32, 5, 1, "same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(  # 第五个卷积
            nn.Conv2d(32, 64, 5, 1, "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(  # 第六个卷积
            nn.Conv2d(64, 128, 3, 1, "same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv7 = nn.Sequential(  # 第七个卷积
            nn.Conv2d(128, 64, 3, 1, "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv8 = nn.Sequential(  # 第八个卷积
            nn.Conv2d(64, 32, 3, 1, "same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv9 = nn.Sequential(  # 第九个卷积
            nn.Conv2d(32, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x


x = torch.randn(10, 1, 64, 64)  # x
y = torch.randn(10, 16, 64, 64)  # y
cnn = CNN()
print(cnn)
print(cnn(x).shape)

# hyper parameters
EPOCH = 1  # 迭代次数
BATCH_SIZE = 5  # 每一次迭代训练的数量
LR = 0.01

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 优化器
loss_func = nn.CrossEntropyLoss()  # 损失函数，交叉熵

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # true为训练时打乱顺序
)  # 将数据拆分成一小块一小块

for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):  # step为总数据/BATCH_SIZE
        print('Epoch:', epoch, '| Step:', step, '| batch x: ',
              batch_x.shape, '| batch y: ', batch_y.shape)
        output = cnn(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()  # 每次把梯度更新为0
        loss.backward()
        optimizer.step()


