import os
import os.path as osp
import random


def split_2ds(data_path, split):
    images_path = []
    labels_path = []
    image_root = osp.join(data_path, 'images')

    for curr_path, sec_paths, _ in os.walk(image_root):
        # print(curr_path)
        for sec_path in sec_paths:
            sec_path = osp.join(curr_path, sec_path)
            # print(sec_path)
            for _, _, files_name in os.walk(sec_path):
                for file_name in files_name:
                    file_path = osp.join(sec_path, file_name)
                    images_path.append(file_path)
                    labels_path.append(file_path.replace('images', 'labels'))
    total_size = len(images_path)
    total_index = [i for i in range(total_size)]
    random.shuffle(total_index)

    train_index = total_index[:int(total_size * split)]
    val_index = total_index[int(total_size * split):]
    train_list = []
    val_list = []
    for i, (image, label) in enumerate(zip(images_path, labels_path)):
        if i in train_index:
            train_list.append((image, label))
        if i in val_index:
            val_list.append((image, label))
    train_dict = [{'image': image, 'label': label}
                  for image, label in train_list]
    val_dict = [{'image': image, 'label': label} for image, label in val_list]

    return train_dict, val_dict


if __name__ == '__main__':
    path = r'../Datasets\test'
    train_dict, val_dict = split_2ds(path, 0.8)
    print(train_dict)
    print(val_dict)
