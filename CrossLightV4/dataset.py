import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import random

class DataAugmentation:
    def __init__(self, jitter_strength=0.02, scale_strength=0.1):
        self.jitter_strength = jitter_strength
        self.scale_strength = scale_strength

    def jitter(self, data):
        noise = torch.randn_like(data) * self.jitter_strength
        return data + noise

    def scale(self, data):
        # 检查数据的维度并适应不同的情况
        if data.dim() == 3:  # (batch_size, channels, sequence_length)
            scale_factor = torch.randn(data.size(0), data.size(1), 1).to(data.device) * self.scale_strength + 1.0
        elif data.dim() == 2:  # (batch_size, sequence_length)
            scale_factor = torch.randn(data.size(0), 1).to(data.device) * self.scale_strength + 1.0
        else:
            raise ValueError("Unexpected data dimensions: {}".format(data.dim()))
        return data * scale_factor

    def weak_augmentation(self, data):
        data = self.jitter(data)
        data = self.scale(data)
        return data

    def strong_augmentation(self, data):
        data = self.jitter(data)
        data = self.scale(data)
        if data.dim() == 3:  # (batch_size, channels, sequence_length)
            perm = torch.randperm(data.size(2))
            data = data[:, :, perm]
        elif data.dim() == 2:  # (batch_size, sequence_length)
            perm = torch.randperm(data.size(1))
            data = data[:, perm]
        else:
            raise ValueError("Unexpected data dimensions: {}".format(data.dim()))
        return data


class TSDataset(Dataset):
    def __init__(self, student_root, student_transform=None):
        self.student_root = student_root
        self.student_transform = student_transform
        self.student_files = []
        self.labels = []
        self.label_map = {}
        self._prepare_dataset()
        self.augmentor = DataAugmentation()

    def _prepare_dataset(self):
        student_classes = sorted(os.listdir(self.student_root))

        for label, cls in enumerate(student_classes):
            student_dir = os.path.join(self.student_root, cls)
            if os.path.isdir(student_dir):
                self.label_map[cls] = label
                student_files = sorted(
                    [os.path.join(student_dir, f) for f in os.listdir(student_dir) if f.endswith('.pt')]
                )
                self.student_files.extend(student_files)
                self.labels.extend([cls] * len(student_files))

    def __len__(self):
        return len(self.student_files)

    def __getitem__(self, idx):
        student_file = self.student_files[idx]
        label = self.labels[idx]
        label_id = self.label_map[label]

        student_data = torch.load(student_file)
        if self.student_transform:
            student_data = self.student_transform(student_data)
            #student_data = self.augmentor.weak_augmentation(student_data)

        return student_data, label, label_id

def Load_data(student_root, batch_size, device='cuda', shuffle=True):

    student_transform = transforms.Compose([
        lambda x: x.float(),
        #normalize
    ])

    dataset = TSDataset(student_root, student_transform=student_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader

# # 示例用法
# student_root = r'G:\CrosslightData705\111'
# batch_size = 64
#
# data_loader = Load_data(student_root, batch_size, shuffle=False)
#
# for batch_data, batch_labels, batch_label_ids in data_loader:
#     batch_data = batch_data
#     batch_label_ids = batch_label_ids
#
#     # 打印一些示例数据用于验证
#     print(f"Batch data shape: {batch_data.shape}")
#     print(f"Batch labels: {batch_labels}")


