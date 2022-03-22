from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def default_loader(path):
    return Image.open(path).crop((728, 885, 2376, 1685)).convert('RGB')


class D2Dataset(Dataset):
    def __init__(self, data_path, trans):
        super().__init__()
        self.data_path = data_path
        self.transform = trans

    def __getitem__(self, index):
        image_path = self.data_path[index]['image']
        label_path = self.data_path[index]['label']
        image = default_loader(image_path)
        label = Image.open(label_path).crop((728, 885, 2376, 1685)).convert('L')
        if self.transform is not None:
            # 数据标签转换为Tensor
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.data_path)


train_transforms = transforms.Compose([
    transforms.Resize((400, 824)),
    transforms.ColorJitter(0.5),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((400, 824)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])
