"""
去雨推理脚本
功能：加载训练好的最佳模型，对测试图像去雨，保存结果为mat文件和png图像
"""

import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
import os
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ============================================
# 网络定义（和训练时一样）
# ============================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out


class DeRainNet(nn.Module):
    def __init__(self, ini_channel=3, channel=32, ResBlock_number=8):
        super(DeRainNet, self).__init__()

        self.conv1 = nn.Conv2d(ini_channel, channel, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bigresblock1 = nn.Sequential(
            *[ResBlock(channel) for _ in range(ResBlock_number)]
        )

        self.bigresblock2 = nn.Sequential(
            *[ResBlock(channel) for _ in range(ResBlock_number)]
        )

        self.bigresblock3 = nn.Sequential(
            *[ResBlock(channel) for _ in range(ResBlock_number)]
        )

        self.upsample2 = UpsampleBlock(channel, scale_factor=2)
        self.upsample4 = UpsampleBlock(channel, scale_factor=4)

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
# 主函数
# ============================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 路径设置
    model_path = './best_model.pth'
    test_path = './test12_chunks.mat'
    output_mat_path = './derained_results.mat'
    output_img_dir = './results'

    # 检查文件
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行训练脚本训练模型")
        return

    if not os.path.exists(test_path):
        print(f"错误：找不到测试数据 {test_path}")
        return

    # 创建输出文件夹
    os.makedirs(output_img_dir, exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    model = DeRainNet(ini_channel=3, channel=32, ResBlock_number=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("模型加载完成")

    # 加载测试数据
    print("正在加载测试数据...")
    test_data = sio.loadmat(test_path)
    img_np = test_data['img']  # [N, H, W, C]
    gt_np = test_data['gt']
    print(f"测试图像数量: {img_np.shape[0]}")
    print(f"图像形状: {img_np.shape}")

    # 存储去雨结果
    derained_list = []

    # 逐张处理
    print("正在去雨...")
    with torch.no_grad():
        for i in range(img_np.shape[0]):
            # 取出单张图像 [H, W, C]
            img_single = img_np[i]

            # 转换格式：[H, W, C] -> [1, C, H, W]
            img_tensor = torch.from_numpy(img_single).float()
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            # 归一化到0-1
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0

            img_tensor = img_tensor.to(device)

            # 去雨
            output = model(img_tensor)
            output = torch.clamp(output, 0, 1)

            # 转回numpy：[1, C, H, W] -> [H, W, C]
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # 转回0-255范围
            output_np_255 = (output_np * 255).astype(np.uint8)

            derained_list.append(output_np_255)

            # 保存去雨后的图像
            img_pil = Image.fromarray(output_np_255)
            img_pil.save(f'{output_img_dir}/derained_{i + 1}.png')

            # 保存原始带雨图像
            rainy_img = img_single
            if rainy_img.max() <= 1.0:
                rainy_img = (rainy_img * 255).astype(np.uint8)
            else:
                rainy_img = rainy_img.astype(np.uint8)
            Image.fromarray(rainy_img).save(f'{output_img_dir}/rainy_{i + 1}.png')

            # 保存Ground Truth
            gt_img = gt_np[i]
            if gt_img.max() <= 1.0:
                gt_img = (gt_img * 255).astype(np.uint8)
            else:
                gt_img = gt_img.astype(np.uint8)
            Image.fromarray(gt_img).save(f'{output_img_dir}/gt_{i + 1}.png')

            print(f'  图像 {i + 1}/{img_np.shape[0]} 处理完成: rainy_{i + 1}.png, derained_{i + 1}.png, gt_{i + 1}.png')

    # 保存为mat文件
    derained_array = np.stack(derained_list, axis=0)  # [N, H, W, C]
    sio.savemat(output_mat_path, {
        'derained': derained_array,
        'original': img_np,
        'gt': gt_np
    })

    print(f'\n结果已保存:')
    print(f'  - MAT文件: {output_mat_path}')
    print(f'  - PNG图像: {output_img_dir}/')
    print('完成!')


if __name__ == '__main__':
    main()