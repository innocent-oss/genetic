import sys
sys.path.append(r'./genetic/code/train/train_model.py')
def get_datasets(dataset_name, data_root, train, transforms=None):
    if dataset_name == 'DRIVE':
        sys.path.append('./genetic/code/dataset')
        from DRIVE_dataset import DRIVE_dataset
        dataset = DRIVE_dataset(data_root=data_root, train=train, transforms=transforms)
        num_return = dataset.num_return
    else:
        raise NotImplementedError

    return dataset, num_return
