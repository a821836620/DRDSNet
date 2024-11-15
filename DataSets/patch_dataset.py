
import random
import torch
import torch.utils.data as data
import math
from skimage.transform import resize
from DataSet.ImageUtils import *
from scipy.ndimage import zoom
from collections import Counter
# from batchgenerators.transforms import Compose
# from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
# from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
#     BrightnessTransform, ContrastAugmentationTransform
# from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
# from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
# def get_train_transform():
#     tr_transforms = []
#
#     tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
#     tr_transforms.append(
#         GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
#                               p_per_sample=0.2, data_key="image"))
#     tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
#     tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
#     tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
#     tr_transforms.append(
#         SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
#                                        order_upsample=3, p_per_sample=0.25,
#                                        ignore_axes=None, data_key="image"))
#     tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
#                                         p_per_sample=0.15, data_key="image"))
#
#     # now we compose these transforms together
#     tr_transforms = Compose(tr_transforms)
#     return tr_transforms

class PatchTrainFolder(data.Dataset):
    def __init__(self, args, crop_size=(64, 192, 192), scale = True):
        self.root = args.root
        self.scale = scale
        self.classes = args.num_classes
        self.files = read_data_paths(self.root, 'train')
        self.crop_d, self.crop_h, self.crop_w = crop_size
        # self.transform = get_train_transform()

    def locate_bbx(self, label, scaler, bbox, cut_rand=0.7):
        scaler_d = int(self.crop_d*scaler)
        scaler_w = int(self.crop_w*scaler)
        scaler_h = int(self.crop_h*scaler)
        img_d, img_h, img_w = label.shape
        boud_d, boud_h, boud_w, = bbox
        margin = 0  # pixels
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scaler_h:
            bbx_h_maxt = bbx_h_max + math.ceil((scaler_h - (bbx_h_max - bbx_h_min)) / 2)
            bbx_h_mint = bbx_h_min - math.ceil((scaler_h - (bbx_h_max - bbx_h_min)) / 2)
            if bbx_h_max >= img_h:
                bbx_h_mint -= (bbx_h_max-img_h)
                bbx_h_maxt = img_h
            if bbx_h_mint < 0:
                bbx_h_maxt -= bbx_h_mint
                bbx_h_mint = 0
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint

        if (bbx_w_max - bbx_w_min) <= scaler_w:
            bbx_w_maxt = bbx_w_max + math.ceil((scaler_w - (bbx_w_max - bbx_w_min)) / 2)
            bbx_w_mint = bbx_w_min - math.ceil((scaler_w - (bbx_w_max - bbx_w_min)) / 2)
            if bbx_w_maxt > img_w:
                bbx_w_mint -= (bbx_w_maxt - img_w)
                bbx_w_maxt = img_w
            if bbx_w_mint < 0:
                bbx_w_maxt -= bbx_w_mint
                bbx_w_mint = 0
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint

        if (bbx_d_max - bbx_d_min) <= scaler_d:
            bbx_d_maxt = bbx_d_max + math.ceil((scaler_d - (bbx_d_max - bbx_d_min)) / 2)
            bbx_d_mint = bbx_d_min - math.ceil((scaler_d - (bbx_d_max - bbx_d_min)) / 2)
            if bbx_d_maxt > img_d:
                bbx_d_mint -= (bbx_d_maxt - img_d)
                bbx_d_maxt = img_d
            if bbx_d_mint < 0:
                bbx_d_maxt -= bbx_d_mint
                bbx_d_mint = 0
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint

        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        if random.random() < cut_rand: # label范围内随机取
            d0 = random.randint(bbx_d_min, np.max([bbx_d_max - scaler_d, bbx_d_min]))
        else:
            d0 = random.randint(0, img_d - scaler_d) # 整个图上随机取
        if random.random() < cut_rand:
            h0 = random.randint(bbx_h_min, np.max([bbx_h_max - scaler_h, bbx_h_min]))
        else:
            h0 = random.randint(0, img_h - scaler_h)
        if random.random() < cut_rand:
            w0 = random.randint(bbx_w_min, np.max([bbx_w_max - scaler_w, bbx_w_min]))
        else:
            w0 = random.randint(0, img_w - scaler_w)
        d1 = d0 + scaler_d
        h1 = h0 + scaler_h
        w1 = w0 + scaler_w
        return [d0, d1, h0, h1, w0, w1]

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = read_datafromITK(datafiles["image_path"])
        label = read_datafromITK(datafiles["label_path"])
        label[label >= self.classes] = self.classes - 1  # 去肿瘤做单器官分割
        if self.scale and np.random.uniform() < 0.5:
            scaler = np.random.uniform(0.7, 1.4)
        else:
            scaler = 1
        if label.shape[0]==label.shape[1] and label.shape[1] != label.shape[2]:
            image = np.transpose(image, (2,0,1))
            label = np.transpose(label, (2,0,1))
        if self.classes>2 and random.random()>0.5:
            bbox = datafiles["bbox"][1]
        else:
            bbox = datafiles["bbox"][0]
        [d0, d1, h0, h1, w0, w1] = self.locate_bbx(label, scaler, bbox)
        image = image[d0: d1, h0: h1, w0: w1]
        label = label[d0: d1, h0: h1, w0: w1]
        if label.shape[0] != self.crop_d or label.shape[1] != self.crop_h or label.shape[2] != self.crop_w:
            image = zoom(image,
                         (self.crop_d / image.shape[0], self.crop_h / image.shape[1], self.crop_w / image.shape[2]),
                         order=1)
            label = zoom(label,
                         (self.crop_d / label.shape[0], self.crop_h / label.shape[1], self.crop_w / label.shape[2]),
                         order=0)
            # 避免缩放时小目标标签丢失。
            # label = np.zeros(new_label.shape, dtype=np.uint8)
            # label[new_label>0] = 1
            # label[new_label>127] = 2
            # print(datafiles["name"],Counter(new_label.flatten()))
            # print(datafiles["name"],Counter(label.flatten()))

        # if not os.path.exists('/home/hjz/data/Duke_breast_tumor/cases/show'): os.makedirs('/home/hjz/data/Duke_breast_tumor/cases/show')
        # itk_image = sitk.ReadImage(datafiles["image_path"], sitk.sitkInt16)
        # save_image = sitk.GetImageFromArray(image)
        # save_image.SetSpacing(itk_image.GetSpacing())
        # save_image.SetOrigin(itk_image.GetOrigin())
        # save_image.SetDirection(itk_image.GetDirection())
        # sitk.WriteImage(save_image, '/home/hjz/data/Duke_breast_tumor/cases/show/image_%s.nrrd'%datafiles['name'])
        #
        # save_label = sitk.GetImageFromArray(label)
        # save_label.SetSpacing(itk_image.GetSpacing())
        # save_label.SetOrigin(itk_image.GetOrigin())
        # save_label.SetDirection(itk_image.GetDirection())
        # sitk.WriteImage(save_label, '/home/hjz/data/Duke_breast_tumor/cases/show/label_%s.nrrd' % datafiles['name'])

        image = image.astype(np.float32)
        # percentile_99_5 = np.percentile(image, 99.5)
        # percentile_00_5 = np.percentile(image, 0.5)
        # image = np.clip(image, percentile_00_5, percentile_99_5)
        # max_n = np.max(image)
        # min_n = np.min(image)
        # image = (image-min_n)/(max_n-min_n)
        image = torch.FloatTensor(image).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)

        return image, label.squeeze(0), datafiles["name"]

    def __len__(self):
        return len(self.files)

class PatchValFolder(data.Dataset):
    def __init__(self, args, crop_size=(64, 192, 192)):
        self.root = args.root
        self.files = read_data_paths(self.root, 'val')
        self.classes = args.num_classes
        self.crop_d, self.crop_h, self.crop_w = crop_size

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = read_datafromITK(datafiles["image_path"])
        label = read_datafromITK(datafiles["label_path"])
        label[label >= self.classes] = self.classes - 1  # 去肿瘤做单器官分割
        if label.shape[0]==label.shape[1] and label.shape[1] != label.shape[2]:
            image = np.transpose(image, (2,0,1))
            label = np.transpose(label, (2,0,1))
        images = []
        for dep in range(0, label.shape[0], self.crop_d):
            for row in range(0, label.shape[1], self.crop_h):
                for col in range(0, label.shape[2], self.crop_w):
                    subimg = image[dep:dep+self.crop_d, row:row+self.crop_h, col:col+self.crop_w]
                    # subimg = np.clip((subimg - 1200) / 1300, -1, 1)  # -1~1 区间
                    if subimg.shape[0] != self.crop_d or subimg.shape[1] != self.crop_h or subimg.shape[2] != self.crop_w:
                        subimg = zoom(subimg,
                                     (self.crop_d / image.shape[0], self.crop_h / image.shape[1],
                                      self.crop_w / image.shape[2]),
                                     order=1)
                    subimg = torch.FloatTensor(subimg).unsqueeze(0)
                    images.append(subimg)
        label =  torch.FloatTensor(label).unsqueeze(0)
        return images, label.squeeze(0), datafiles['name']

    def __len__(self):
        return len(self.files)

class PatchPredFolder(data.Dataset):
    def __init__(self, args, crop_size=(64, 192, 192)):
        self.root = args.root
        self.files = read_pred_data(self.root, 'test')
        self.classes = args.num_classes
        self.crop_d, self.crop_h, self.crop_w = crop_size
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = read_datafromITK(datafiles["image_path"])
        label = read_datafromITK(datafiles["label_path"])
        images = []
        for dep in range(0, label.shape[0], self.crop_d):
            for row in range(0, label.shape[1], self.crop_h):
                for col in range(0, label.shape[2], self.crop_w):
                    subimg = image[dep:dep+self.crop_d, row:row+self.crop_h, col:col+self.crop_w]
                    # subimg = np.clip((subimg - 1200) / 1300, -1, 1)  # -1~1 区间
                    if subimg.shape[0] != self.crop_d or subimg.shape[1] != self.crop_h or subimg.shape[2] != self.crop_w:
                        subimg = zoom(subimg,
                                     (self.crop_d / image.shape[0], self.crop_h / image.shape[1],
                                      self.crop_w / image.shape[2]),
                                     order=1)
                    subimg = torch.FloatTensor(subimg).unsqueeze(0)
                    images.append(subimg)
        return images, datafiles['name']

    def __len__(self):
        return len(self.files)