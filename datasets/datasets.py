from pathlib import Path
import torch
import torchvision

from transformations.simclr import TransformsSimCLR
from datasets.inaturalist_2019 import INaturalist2019Dataset


class Datasets:

    @staticmethod
    def get_simclr_dataset(config):
        if config.simclr.train.dataset == "STL10":

            train_dataset = torchvision.datasets.STL10(
                config.simclr.train.data_dir_path,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=config.simclr.train.img_size)
            )

            classes = train_dataset.classes

        elif config.simclr.train.dataset == "CIFAR10":

            train_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=config.simclr.train.img_size)
            )

            classes = train_dataset.classes

        elif config.simclr.train.dataset == 'ImageNet':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('ImageNet')
            train_dataset = torchvision.datasets.ImageNet(dataset_dir_path, split='train',
                                                          transform=TransformsSimCLR(size=config.simclr.train.img_size))
            classes = train_dataset.classes

        elif config.simclr.train.dataset == 'iNaturalist2019':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('inaturalist-2019-fgvc6')

            train_json_file_path = dataset_dir_path.joinpath('train2019_0.2.json')
            train_dataset = INaturalist2019Dataset(train_json_file_path, dataset_dir_path,
                                                   transform=TransformsSimCLR(size=config.simclr.train.img_size))

            classes = train_dataset.classes
        else:
            raise NotImplementedError('invalid dataset: {}'.format(config.simclr.train.dataset))

        return train_dataset, classes

    @staticmethod
    def get_datasets(config):
        if config.simclr.train.dataset == "STL10":

            train_dataset = torchvision.datasets.STL10(
                config.simclr.train.data_dir_path,
                split="train",
                download=True,
                transform=TransformsSimCLR(size=config.simclr.train.img_size).train_test_transform,
            )

            classes = train_dataset.classes

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                       [int(0.8 * len(train_dataset)),
                                                                        int(0.2 * len(train_dataset))])
            test_dataset = torchvision.datasets.STL10(
                config.simclr.train.data_dir_path,
                split="test",
                download=True,
                transform=TransformsSimCLR(config.simclr.train.img_size).test_transform,
            )
        elif config.simclr.train.dataset == "CIFAR10":

            train_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=config.simclr.train.img_size).train_test_transform,
            )

            classes = train_dataset.classes

            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                       [int(0.8 * len(train_dataset)),
                                                                        int(0.2 * len(train_dataset))])
            test_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=False,
                download=True,
                transform=TransformsSimCLR(config.simclr.train.img_size).test_transform,
            )

        elif config.simclr.train.dataset == 'ImageNet':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('ImageNet')
            train_dataset = torchvision.datasets.ImageNet(dataset_dir_path, split='train',
                                                          transform=TransformsSimCLR(
                                                              size=config.simclr.train.img_size).train_test_transform)

            classes = train_dataset.classes

            val_dataset = torchvision.datasets.ImageNet(dataset_dir_path, split='val',
                                                        transform=TransformsSimCLR(
                                                            size=config.simclr.train.img_size).test_transform)

            # no labeled test data is available
            test_dataset = val_dataset

        elif config.simclr.train.dataset == 'iNaturalist2019':

            data_dir_path = Path(config.simclr.train.data_dir_path)
            dataset_dir_path = data_dir_path.joinpath('inaturalist-2019-fgvc6')

            train_json_file_path = dataset_dir_path.joinpath('train2019_0.2.json')
            train_dataset = INaturalist2019Dataset(train_json_file_path, dataset_dir_path,
                                                   transform=TransformsSimCLR(
                                                       size=config.simclr.train.img_size).train_test_transform)

            classes = train_dataset.classes

            val_json_file_path = dataset_dir_path.joinpath('val2019_0.2.json')
            val_dataset = INaturalist2019Dataset(val_json_file_path, dataset_dir_path,
                                                 transform=TransformsSimCLR(
                                                     size=config.simclr.train.img_size).test_transform)
            # no labeled test data is available
            test_dataset = val_dataset

        else:
            raise NotImplementedError('invalid dataset: {}'.format(config.simclr.train.dataset))

        return train_dataset, val_dataset, test_dataset, classes

    @staticmethod
    def get_simclr_loader(config, train_dataset):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.simclr.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
            sampler=None,
        )

        return train_loader

    @staticmethod
    def get_loaders(config, train_dataset, val_dataset, test_dataset):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.simclr.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.onnx.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.logistic_regression.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
        )

        return train_loader, val_loader, test_loader
