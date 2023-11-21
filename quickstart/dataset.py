import mindspore
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from download import download

# Download data from open datasets
# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
#       "notebook/datasets/MNIST_Data.zip"
# path = download(url, "./", kind="zip", replace=True)

def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

if __name__ == "__main__":

    train_dataset = MnistDataset('MNIST_Data/train')
    test_dataset = MnistDataset('MNIST_Data/test')
    print(train_dataset.get_col_names())

    # Map vision transforms and batch dataset
    train_dataset = datapipe(train_dataset, 64)
    test_dataset = datapipe(test_dataset, 64)

    for image, label in test_dataset.create_tuple_iterator():
        print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
        print(f"Shape of label: {label.shape} {label.dtype}")
        break

    for data in test_dataset.create_dict_iterator():
        print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
        print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
        break
