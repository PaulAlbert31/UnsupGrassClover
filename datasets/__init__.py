from mypath import Path

def irish(red=False, root=Path.db_root_dir('irish'), transform=None, transform_test=None):
    from datasets.irish import make_dataset, Irish
    train_data, train_labels, val_data, val_labels = make_dataset(root=root, red=red)
    trainset = Irish(train_data, train_labels, transform=transform)
    if False:
        from datasets.irish_phone import make_dataset as make_dataset_phone
        _, _, val_data, val_labels = make_dataset_phone(root=Path.db_root_dir('irish_phone'))    
    testset = Irish(val_data, val_labels, transform=transform_test)
    return trainset, testset

def danish(root=Path.db_root_dir('danish'), transform=None, transform_test=None):
    from datasets.danish import make_dataset as make_dataset_danish
    from datasets.danish import Danish
    train_data, train_labels, val_data, val_labels = make_dataset_danish(root=root)
    trainset = Danish(train_data, train_labels, transform=transform)
    testset = Danish(val_data, val_labels, transform=transform_test)
    return trainset, testset

def irish_iMix(root=Path.db_root_dir('irish_unsup'), transform=None, transform_test=None):
    from datasets.irish_unsup import Clover_Unsup, make_dataset_clover, make_dataset_clover_unlab
    train_data = make_dataset_clover(root=Path.db_root_dir('irish'))
    train_data_unlab = make_dataset_clover_unlab(root=root)
    trainset = Clover_Unsup(train_data + train_data_unlab, transform=transform)
    return trainset, None

def danish_iMix(root=Path.db_root_dir('danish_unsup'), transform=None, transform_test=None):
    from datasets.danish_unsup import Danish_Unsup, make_dataset_clover
    train_data, train_labels, val_data, val_labels = make_dataset_clover(root=root)
    trainset = Danish_Unsup(train_data, train_labels, train=True, transform=transform)
    return trainset, None
