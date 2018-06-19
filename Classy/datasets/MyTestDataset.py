import torch.utils.data as data
from PIL import Image
import os
import os.path
import csv
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)
def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    # for root, _, fnames in sorted(os.walk(dir)):
    for root, _, fnames in os.walk(dir):
        # for fname in sorted(fnames):
        for fname in fnames:
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images
def read_list(LIST_PATH):
    if os.path.exists(LIST_PATH):
        list = []
        with open(LIST_PATH, 'r') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] !='FILE_ID':
                    list.append(row[0])
        return list
    else:
        print('No {} file'.format(LIST_PATH))

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def check_csv_image(list, data_root):
    n = 0
    error = 0
    for name in list:
        if any(os.path.exists(os.path.join(data_root, name + e)) for e in IMG_EXTENSIONS):
            n = n +1
        else:
            error = error + 1
            print('filename : {} is not exist'.format(name))
    if len(list) == n:
        print('Total check {} files, and all exist\n\r'.format(n))
        return True
    else:
        print('Total check {} files, and error {} files\n\r'.format(n, error))
        return False


class MyTestDataset(data.Dataset):

    def __init__(self, data_root, csv_path, loader=default_loader, extensions=IMG_EXTENSIONS, transform=None, target_transform=None):
        list = read_list(csv_path)
        samples = make_dataset(data_root, extensions)
        if check_csv_image(list, data_root) and len(list)==len(samples):

            self.data_root = data_root
            self.loader = loader
            self.extensions = extensions
            self.samples = samples



            self.transform = transform
            self.target_transform = target_transform
        else:
            print('Can not create database !')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path
    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

