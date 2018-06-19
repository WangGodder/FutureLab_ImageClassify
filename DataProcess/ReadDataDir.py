import torch.utils.data as data
import csv
from PIL import Image
import os
import os.path
import shutil
import numpy as np
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile))



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB').resize((224,224))

def create_image_dir(dir_name, list_path, categories_path, images_dir, rate, postfix):
    DIR_PATH = os.path.join('.', dir_name)
    LIST_PATH = os.path.join('.',list_path)
    CATEGORIES_PATH = os.path.join('.', categories_path)
    if not os.path.exists(LIST_PATH) or not os.path.exists(CATEGORIES_PATH):
        print("No List File（CSV）")
        return

    if os.path.exists(DIR_PATH) and os.path.exists(LIST_PATH):
        shutil.rmtree(DIR_PATH)
        os.mkdir(DIR_PATH)
        os.mkdir(os.path.join(DIR_PATH, 'train'))
        os.mkdir(os.path.join(DIR_PATH, 'test'))

    else:
        os.mkdir(DIR_PATH)
        os.mkdir(os.path.join(DIR_PATH, 'train'))
        os.mkdir(os.path.join(DIR_PATH, 'test'))
    list = []
    label_diff = []
    list_categories = []
    list_categories_name = []
    with open(CATEGORIES_PATH, 'r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[0].isdigit():
                list_categories.append(row[0])
                list_categories_name.append(row[1])


    with open(LIST_PATH, 'r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[1].isdigit():
                list.append(row)
                if row[1] not  in label_diff:
                    label_diff.append(row[1])

                #print(row[0])
                #print(row[1])
                #print(type(row))  # 类型为一个list
    print(len(list))
    print(len(label_diff))
    list_classied = []

    for label in range(0,len(label_diff)):
        list_classied_temp = []
        for image in list:
            if int(image[1]) == label:
                image[0] = image[0] + postfix
                list_classied_temp.append(image)
        list_classied.append(list_classied_temp)
        list_train = []
        list_test = []

    for class_index in list_classied:
        #print(list_categories_name[list_categories.index(class_index[0][1])])
        name = list_categories_name[list_categories.index(class_index[0][1])]
        os.mkdir(os.path.join(DIR_PATH, 'train', name))
        os.mkdir(os.path.join(DIR_PATH, 'test', name))
        num = int(len(class_index) * rate)
        for i in range(0,num):
            mycopyfile(os.path.join(images_dir,class_index[i][0]),os.path.join(DIR_PATH, 'train', name, class_index[i][0]))
            #print(os.path.join(images_dir,class_index[i][0]))
            #print(os.path.join(DIR_PATH, 'train',class_index[i][0]))
        for i in range(num,len(class_index)):
            mycopyfile(os.path.join(images_dir, class_index[i][0]), os.path.join(DIR_PATH, 'test',name, class_index[i][0]))
            #print(os.path.join(images_dir, class_index[i][0]))
            #print(os.path.join(DIR_PATH, 'train', class_index[i][0]))

    with open(os.path.join(DIR_PATH, 'train','train_list.csv'), 'w', newline='') as csvfile:  # 解决写入空行问题 使用wb不会再每一行后面插入空行
        csvwriter = csv.writer(csvfile)
        lst = [[1, 2, 3], [4, 5, 6]]
        csvwriter.writerows(lst)



    #print(len(list_classied))
    #print(len(list_classied[0]))
    #print(len(list_classied[0][0]))







class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    create_image_dir('data',
                     'list.csv',
                     'categories.csv',
                     os.path.join('.', 'images','data'),
                     0.8,
                     '.jpg')

