from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


class D2Dataset(Dataset):
    def __init__(self, data_path, trans):
        super().__init__()
        self.data_path = data_path
        self.transform = trans

    def __getitem__(self, index):
        image_path = self.data_path[index][0]
        label_path = self.data_path[index][1]
        image = default_loader(image_path)
        label = default_loader(label_path)
        if self.transform is not None:
            # 数据标签转换为Tensor
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.data_path)


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])
