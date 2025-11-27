from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from data.fewshot_datasets import *

ID_to_DIRNAME={
    'imagenet': 'domain_shift_datastes/imagenet/images/val',
    'imagenet_a': 'domain_shift_datastes/imagenet-a/imagenet-a',
    'imagenet_sketch': 'domain_shift_datastes/ImageNet-Sketch/images',
    'imagenet_r': 'domain_shift_datastes/imagenet-r/imagenet-r',
    'imagenetv2': 'domain_shift_datastes/imagenetv2-matched-frequency-format-val/imagenetv2-matched-frequency-format-val',
    'imagenet_c': 'corruption/imagenet-c',
    'oxford_flowers': 'fine-grained/oxford_flowers',
    'dtd': 'fine-grained/dtd',
    'oxford_pets': 'fine-grained/oxford_pets',
    'stanford_cars': 'fine-grained/stanford_cars',
    'ucf101': 'fine-grained/ucf101',
    'caltech101': 'fine-grained/caltech-101',
    'food101': 'fine-grained/food-101',
    'sun397': 'fine-grained/sun397',
    'fgvc_aircraft': 'fine-grained/fgvc_aircraft',
    'eurosat': 'fine-grained/eurosat',
}



def build_test_loader(set_id, transform, data_root, batch_size, corruption=None, level=None):
    if set_id in ['imagenet_a', 'imagenet_sketch', 'imagenet_r', 'imagenetv2']:
        transform = get_ood_preprocess(63)
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id =='imagenet':
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id =='imagenet_c':
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id], corruption, level)
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform)
    else:
        raise NotImplementedError
    val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return val_loader


# Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def aug(image, preprocess):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    return x_processed


class Augmenter(object):
    def __init__(self, base_transform, preprocess, n_views=63):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        if self.n_views == 0:
            return image
        else:
            views = [aug(x, self.preprocess) for _ in range(self.n_views)]
            return [image] + views

def get_ood_preprocess(num_views):
    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])

    data_transform = Augmenter(base_transform, preprocess, n_views=num_views) # w/O mix_augmentation

    return data_transform