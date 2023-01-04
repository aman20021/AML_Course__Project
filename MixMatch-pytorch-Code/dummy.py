import torchvision.transforms as transforms
import dataset.cifar10 as dataset

transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

transform_val = transforms.Compose([
    dataset.ToTensor(),
])

train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10('./data', 250, transform_train=transform_train, transform_val=transform_val)