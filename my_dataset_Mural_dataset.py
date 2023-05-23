import os
import cv2 as cv
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import functional as F


class DatasetTrain(data.Dataset):
    def __init__(self, dataset_root, transforms=None,):
        super(DatasetTrain, self).__init__()
        root = os.path.join(dataset_root)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, "our_Images", "train")
        mask_dir = os.path.join(root, "Binary_our_masks", "train")
        # self.images = glob.glob(os.path.join(image_dir, "*"))
        # self.masks = glob.glob(os.path.join(mask_dir, "*"))
        image_file_names = os.listdir(image_dir)
        mask_file_names = os.listdir(mask_dir)
        self.images = [os.path.join(image_dir, x) for x in image_file_names]    # 图片路径为：D:\TheHuns-code\data\new_mural_dataset\our_Images\train
        self.masks = [os.path.join(mask_dir, x) for x in mask_file_names]       # mask路径为：D:\TheHuns-code\data\new_mural_dataset\our_new_masks\train
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


class DatasetVal(data.Dataset):
    def __init__(self, dataset_root, transforms=None,):
        super(DatasetVal, self).__init__()
        root = os.path.join(dataset_root)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, "our_Images", "val")
        mask_dir = os.path.join(root, "Binary_our_masks", "val")        
        image_file_names = os.listdir(image_dir)
        mask_file_names = os.listdir(mask_dir)
        self.images = [os.path.join(image_dir, x) for x in image_file_names]    # 图片路径为：D:\TheHuns-code\data\new_mural_dataset\our_Images\val
        self.masks = [os.path.join(mask_dir, x) for x in mask_file_names]       # mask路径为：D:\TheHuns-code\data\new_mural_dataset\our_new_masks\val
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        # target = cv.imread(self.masks[index], 0)             # 读取单通道灰度图的方法
        target = Image.open(self.masks[index])  # 读取三通道图像的方法
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
