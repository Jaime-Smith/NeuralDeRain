import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as sio
import h5py
from torch.utils.data import Dataset
import os

from inference import ResBlock

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#====================
#part1: data loading---tested correct
#====================

class TestDatasetMat(Dataset):
    def __init__(self, file_path):
        super(TestDatasetMat, self).__init__()
        self.data = None
        self.file_path = file_path
        data = sio.loadmat(self.file_path)
        self.inputs = torch.from_numpy(data["img"][...])
        self.targets = torch.from_numpy(data["gt"][...])
        print(self.inputs.shape, self.targets.shape)
        self.length = len(self.inputs)

    def __getitem__(self, index):
        index = index % self.length

        return {'img': self.inputs[index, ...], 'gt': self.targets[index, ...]}

    def __len__(self):
        return self.length


class TrainValDatasetH5(Dataset):
    def __init__(self, file_path):
        super(TrainValDatasetH5, self).__init__()
        self.data = None
        self.file_path = file_path
        data = h5py.File(self.file_path, 'r')
        self.inputs = torch.from_numpy(data["img"][...])
        self.targets = torch.from_numpy(data["gt"][...])
        print("img:", self.inputs.shape)
        print("gt:", self.targets.shape)

        self.length = len(self.inputs)

    def __getitem__(self, index):
        index = index % self.length

        return {'img': self.inputs[index, ...], 'gt': self.targets[index, ...]}

    def __len__(self):
        return self.length

# ============================================
# Part2:define blocks
# ============================================

class resblock(nn.Module):
    def __init__(self ):
        global channel
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
    def forward(self, x):
        residual = x  # 保存输入，用于残差连接
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual  # 残差连接：输出 = 卷积结果 + 原始输入
        return out

class UpsampleBlock(nn.Module):
    def __init__(self,scaler):
        global channel
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scaler, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU( inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out

class DeRainNet(nn.Module):
    def __init__(self):
        super(DeRainNet, self).__init__()

        global ini_channel,channel,ResBlock_number
        self.conv1 = nn.Conv2d(ini_channel,channel,3,padding=1)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2,2)
        self.bigresblock1 = nn.Sequential(
            *[resblock() for _ in range(ResBlock_number)]
        )
        self.bigresblock2 = nn.Sequential(
            *[resblock() for _ in range(ResBlock_number)]
        )
        self.bigresblock3 = nn.Sequential(
            *[resblock() for _ in range(ResBlock_number)]
        )
        self.upsample2 = UpsampleBlock( scaler=2)  # 1/2x -> 1x
        self.upsample4 = UpsampleBlock( scaler=4)  # 1/4x -> 1x

        self.decat_conv = nn.Conv2d(channel * 3, channel, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(channel, 3, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x

        root = self.relu(self.conv1(x))
        branch1 = self.bigresblock1(root)

        root_half = self.maxpool(root)
        branch2 = self.bigresblock2(root_half)
        branch2 = self.upsample2(branch2)

        root_quarter = self.maxpool(root_half)
        branch3 = self.bigresblock3(root_quarter)
        branch3 = self.upsample4(branch3)

        concat = torch.cat([branch1, branch2, branch3], dim=1)
        deconcat = self.decat_conv(concat)
        output = self.output_conv(deconcat)

        output = identity - output
        return output

# ============================================
# Part3:Train
# ============================================

def one_epoch(model,train_loader,device):
    model.train()
    total_loss = 0
    for batch_index,batch in enumerate(train_loader):
        img = batch['img'].to(device)
        target = batch['gt'].to(device)
        gt = batch['gt']

        criterion = nn.L1Loss()  # L1 Loss，如图所示
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_index % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_index}/{len(train_loader)}, Loss: {loss.item():.6f}')
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device):
    """验证"""
    model.eval()
    total_loss = 0
    criterion = nn.L1Loss()  # L1 Loss，如图所示


    with torch.no_grad():
        for batch in val_loader:
            img = batch['img'].to(device)
            gt = batch['gt'].to(device)

            output = model(img)
            loss = criterion(output, gt)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss







if __name__ == "__main__":
    # 超参数
    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 100
    device = torch.device('cpu')


    #参数
    channel = 32
    ini_channel = 3
    ResBlock_number = 8

    from torch.utils.data import DataLoader
    print("train_data:")
    train_dataset = TrainValDatasetH5(file_path="./Rain200L_train_chunks.h5")
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=4)
    print("lenth of train_loader:", len(train_loader))

    print("Val_Data:")
    val_dataset = TrainValDatasetH5(file_path="./Rain200L_val_chunks.h5")
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=4)
    print("lenth of val_loader:", len(val_loader))

    model = DeRainNet().to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')

    print("开始训练...")
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f'\n===== Epoch {epoch}/{num_epochs} =====')

        # 训练
        train_loss = one_epoch(model, train_loader, device)
        print(f'训练损失: {train_loss:.6f}')

        val_loss = validate(model, val_loader, device)
        print(f'验证损失: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型，验证损失: {val_loss:.6f}')


    print('训练完成！')






