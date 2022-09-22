import sys
sys.path.append(r'/content/drive/MyDrive/Genetic-U-Net-main(1)/Genetic-U-Net-main/code/train/train_model.py')
def get_datasets(dataset_name, data_root, train, transforms=None):
    if dataset_name == 'DRIVE':
        sys.path.append('/content/drive/MyDrive/Genetic-U-Net-main(1)/Genetic-U-Net-main/code/dataset')
        from DRIVE_dataset import DRIVE_dataset
        dataset = DRIVE_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
    else:
        raise NotImplementedError

    return dataset, num_return
