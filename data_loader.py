''' Digit experiment
'''
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, USPS, SVHN, CIFAR10, STL10, ImageFolder

import tensorflow_datasets as tfds

import os
import pickle
import numpy as np
from scipy.io import loadmat
from PIL import Image

from tools.autoaugment import SVHNPolicy, CIFAR10Policy
from tools.randaugment import RandAugment

HOME = os.environ['HOME']

class myTensorDataset(Dataset):
    def __init__(self, x, y, transform=None, twox=False):
        self.x = x
        self.y = y
        self.transform = transform
        self.twox = twox
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        if self.transform is not None:
            x = self.transform(x)
            if self.twox:
                x2 = self.transform(x)
                return (x, x2), y
        return x, y



def resize_imgs(x, size):
    ''' Only for single channel images 
        x [n, 28, 28]
        size int
    '''
    resize_x = np.zeros([x.shape[0], size, size])
    for i, im in enumerate(x):
        im = Image.fromarray(im)
        im = im.resize([size, size], Image.ANTIALIAS)
        resize_x[i] = np.asarray(im)
    return resize_x

def resize_imgs_dkcho(x,size):
    '''
    For Multi-Channel Images
    '''
    resize_x = np.zeros([x.shape[0], size, size, 3])
    transform= transforms.Resize(size= (size,size))
    for i, im in enumerate(x):
            im = Image.fromarray(im)
            im= transform(im)
            im= np.asarray(im)
            #print(im.shape) #(32, 32, 3)
            #im = im.resize([size, size], Image.ANTIALIAS)
            resize_x[i] = im
    return resize_x
        
       
def load_mnist(split='train', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    '''
        autoaug == 'AA', AutoAugment
                   'FastAA', Fast AutoAugment
                   'RA', RandAugment
        channels == 3 return by default rgb 3 channel image
                    1 Return a single channel image
    '''
    path = f'data/mnist-{split}.pkl'
    if not os.path.exists(path):
        dataset = MNIST(f'{HOME}/.pytorch/MNIST', train=(split=='train'), download=True)
        x, y = dataset.data, dataset.targets
        if split=='train':
            #[TODO]- Why should we only use 10k images?
            #x, y = x, y
            x, y = x[0:10000], y[0:10000]
        x = torch.tensor(resize_imgs(x.numpy(), 32))
        x = (x.float()/255.).unsqueeze(1).repeat(1,3,1,1)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    
    # Data Augmentation Pipeline
    transform = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'AA':
            transform.append(SVHNPolicy())
        elif autoaug == 'RA':
            transform.append(RandAugment(3,4))
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    dataset = myTensorDataset(x, y, transform=transform, twox=twox)
    return dataset

#[TODO] Add cifar10
def load_cifar10(split='train', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    '''
        autoaug == 'AA', AutoAugment
                   'FastAA', Fast AutoAugment
                   'RA', RandAugment
        channels == 3 return by default rgb 3 channel image
                    1 Return a single channel image
    '''
    path = f'data/cifar10-{split}.pkl'
    cifar10_transforms_train= transforms.Compose([transforms.Resize((32,32))]) #224,224
    if not os.path.exists(path):
        dataset = CIFAR10(f'{HOME}/.pytorch/CIFAR10', train=(split=='train'), download=True, transform= cifar10_transforms_train)
        x, y = dataset.data, dataset.targets
        
        #Only Select First 10k images as train
        #if split=='train':
        #    x, y = x[0:10000], y[0:10000]
        
        #[TODO] - solve -> AttributeError: 'numpy.ndarray' object has no attribute 'numpy'
        #x = torch.tensor(resize_imgs(x.numpy(), 32))
        #x = torch.tensor(resize_imgs_dkcho(x, 32)) # x-> torch.Size([10000, 32, 32, 3])
        x= torch.tensor(x)
        x = (x.float()/255.)#.unsqueeze(1).repeat(1,3,1,1)  #<class 'torch.Tensor'>
        x= x.permute(0,3,1,2) #[batchsize,w,h,channel] -> [batchsize, channel, w,h]
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    
    # Data Augmentation Pipeline
    transform = [transforms.ToPILImage()]
    if translate is not None:
        transform.append(transforms.RandomAffine(0, [translate, translate]))
    if autoaug is not None:
        if autoaug == 'AA':
            transform.append(CIFAR10Policy()) #originally SVHNPolicy()
        elif autoaug == 'RA':
            transform.append(RandAugment(3,4))
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)
    dataset = myTensorDataset(x, y, transform=transform, twox=twox)
    return dataset




def load_usps(split='train', channels=3):
    path = f'data/usps-{split}.pkl'
    if not os.path.exists(path):
        dataset = USPS(f'{HOME}/.pytorch/USPS', train=(split=='train'), download=True)
        x, y = dataset.data, dataset.targets
        x = torch.tensor(resize_imgs(x, 32))
        x = (x.float()/255.).unsqueeze(1).repeat(1,3,1,1)
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    dataset = TensorDataset(x, y)
    return dataset

def load_svhn(split='train', channels=3):
    dataset = SVHN(f'{HOME}/.pytorch/SVHN', split=split, download=True)
    x, y = dataset.data, dataset.labels
    x = x.astype('float32')/255.
    x, y = torch.tensor(x), torch.tensor(y)
    if channels == 1:
        x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

def load_syndigit(split='train', channels=3):
    path = f'data/synth_{split}_32x32.mat'
    data = loadmat(path)
    x, y = data['X'], data['y']
    x = np.transpose(x, [3, 2, 0, 1]).astype('float32')/255.
    y = y.squeeze()
    x, y = torch.tensor(x), torch.tensor(y)
    if channels == 1:
        x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

def load_mnist_m(split='train', channels=3):
    path = f'data/mnist_m-{split}.pkl'
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        x, y = torch.tensor(x.astype('float32')/255.), torch.tensor(y)
        if channels==1:
            x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

def load_stl10(split='train', channels=3):
    STL10_transforms_train= transforms.Compose([transforms.Resize((32,32))])
    dataset = STL10(f'{HOME}/.pytorch/STL10', split=split, download=True, transform= STL10_transforms_train)
    x, y = dataset.data, dataset.labels
    x = x.astype('float32')/255.
    x, y = torch.tensor(x), torch.tensor(y)
    if channels == 1:
        x = x.mean(1, keepdim=True)
    dataset = TensorDataset(x, y)
    return dataset

def load_cifar10c(split='train', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    path = f'data/cifar10c-{split}.pkl'
    cifar10_transforms_train= transforms.Compose([transforms.Resize((32,32))]) #224,224
    if not os.path.exists(path):
        dataset = tfds.as_numpy(tfds.load('cifar10_corrupted', split= split, shuffle_files= True, batch_size= -1))
        x, y = dataset['image'], dataset['label']
        x= torch.tensor(x)
        x = (x.float()/255.)#.unsqueeze(1).repeat(1,3,1,1)  #<class 'torch.Tensor'>
        x= x.permute(0,3,1,2) #[batchsize,w,h,channel] -> [batchsize, channel, w,h]
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset

def load_cifar10c_level(split='test', ctype= 'fog', level= 5, translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    path = f'data/cifar10c-{ctype}_{level}.pkl'
    cifar10_transforms_train= transforms.Compose([transforms.Resize((32,32))]) #224,224
    if not os.path.exists(path):
        tfpath= f'cifar10_corrupted/{ctype}_{level}'.format(ctype= ctype, level= level)
        dataset = tfds.as_numpy(tfds.load(tfpath, split= split, shuffle_files= True, batch_size= -1))
        x, y = dataset['image'], dataset['label']
        x= torch.tensor(x)
        x = (x.float()/255.)#.unsqueeze(1).repeat(1,3,1,1)  #<class 'torch.Tensor'>
        x= x.permute(0,3,1,2) #[batchsize,w,h,channel] -> [batchsize, channel, w,h]
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset

def load_cifar10_1(split='train', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    path = f'data/cifar10_1-{split}.pkl'
    cifar10_transforms_train= transforms.Compose([transforms.Resize((32,32))]) #224,224
    if not os.path.exists(path):
        dataset = tfds.as_numpy(tfds.load('cifar10_1', split= split, shuffle_files= True, batch_size= -1))
        x, y = dataset['image'], dataset['label']
        #dataset = CIFAR10(f'{HOME}/.pytorch/CIFAR10', train=(split=='train'), download=True, transform= cifar10_transforms_train)
        #x, y = dataset.data, dataset.targets
        
        #Only Select First 10k images as train
        #if split=='train':
        #    x, y = x[0:10000], y[0:10000]
        
        #[TODO] - solve -> AttributeError: 'numpy.ndarray' object has no attribute 'numpy'
        #x = torch.tensor(resize_imgs(x.numpy(), 32))
        #x = torch.tensor(resize_imgs_dkcho(x, 32)) # x-> torch.Size([10000, 32, 32, 3])
        x= torch.tensor(x)
        x = (x.float()/255.)#.unsqueeze(1).repeat(1,3,1,1)  #<class 'torch.Tensor'>
        x= x.permute(0,3,1,2) #[batchsize,w,h,channel] -> [batchsize, channel, w,h]
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    

#PACS
def load_pacs(split='test', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    #PACS Dataset
    NUM_CLASSES = 7      # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
    DATASETS_NAMES = ['photo', 'art', 'cartoon', 'sketch']
    CLASSES_NAMES = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
    DIR_PHOTO = './data/PACS/photo'
    DIR_ART = './data/PACS/art_painting'
    DIR_CARTOON = './data/PACS/cartoon'
    DIR_SKETCH = './data/PACS/sketch'

    path = f'data/pacs-{split}.pkl'
    #manually use PACS-Photo as train/test
    trainpath= 'data/pacs-train.pkl'
    testpath= 'data/pacs-test.pkl'

    pacs_convertor= {'train':DIR_PHOTO,'test':DIR_PHOTO}
    
    pacs_transforms_train= transforms.Compose([transforms.CenterCrop(224),transforms.Resize((224,224)),transforms.ToTensor()]) #224,224
    if not os.path.exists(path):
        
        dataset= ImageFolder(pacs_convertor[split], transform=pacs_transforms_train)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

        #Train Set
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_size,drop_last=True)
        x, y= next(iter(train_loader))
        x= torch.tensor(x)
        y = torch.tensor(y)
        with open(trainpath, 'wb') as f:
            pickle.dump([x, y], f)
        
        #Test Set
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_size,drop_last=True, shuffle=True)
        x, y= next(iter(test_loader))
        x= torch.tensor(x)
        y = torch.tensor(y)
        with open(testpath, 'wb') as f:
            pickle.dump([x, y], f) 

    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset

def load_pacs_acs(split='art', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    #PACS Dataset
    NUM_CLASSES = 7     
    DATASETS_NAMES = ['photo', 'art', 'cartoon', 'sketch']
    CLASSES_NAMES = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
    DIR_ART = './data/PACS/art_painting'
    DIR_CARTOON = './data/PACS/cartoon'
    DIR_SKETCH = './data/PACS/sketch'
    DIR_PHOTO = './data/PACS/photo'

    path = f'data/pacs-{split}.pkl'
    pacs_convertor= {'photo':DIR_PHOTO, 'art':DIR_ART, 'cartoon':DIR_CARTOON, 'sketch':DIR_SKETCH}
    
    pacs_transforms_train= transforms.Compose([transforms.CenterCrop(224),transforms.Resize((224,224)),transforms.ToTensor()]) #224,224
    
    if not os.path.exists(path):
        
        dataset= ImageFolder(pacs_convertor[split], transform=pacs_transforms_train)
        #train_size = int(0.8 * len(dataset))
        #test_size = len(dataset) - train_size
        #train_set, test_set = random_split(dataset, [train_size, test_size])
        
        #Test Set
        test_size= len(dataset)
        test_loader = torch.utils.data.DataLoader(dataset,batch_size=test_size,drop_last=True, shuffle=True)
        x, y= next(iter(test_loader))
        x= torch.tensor(x)
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f) 
    
    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
        
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset

def load_officehome(split='train', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    DIR_REALWORLD = './data/office_home/Real World'
    trainpath= 'data/officehome-train.pkl'
    testpath= 'data/officehome-test.pkl'
    path = f'data/officehome-{split}.pkl'
    officehome_convertor= {'train':trainpath, 'test':testpath}
    officehome_transforms_train= transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]) #224,224
    if not os.path.exists(path):
        
        dataset= ImageFolder(DIR_REALWORLD, transform=officehome_transforms_train)
        
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])
        
        #Train Set
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=train_size,drop_last=True)
        x, y= next(iter(train_loader))
        x= torch.tensor(x)
        y = torch.tensor(y)
        with open(trainpath, 'wb') as f:
            pickle.dump([x, y], f)
        
        #Test Set
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_size,drop_last=True, shuffle=True)
        x, y= next(iter(test_loader))
        x= torch.tensor(x)
        y = torch.tensor(y)
        with open(testpath, 'wb') as f:
            pickle.dump([x, y], f) 

    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset
    
def load_officehome_domain(domain='Product', translate=None, twox=False, ntr=None, autoaug=None, channels=3):
    DATASETS_NAMES = ['Art', 'Clipart', 'Product', 'Real World']
    DIR_Art = './data/office_home/Art'
    DIR_Clipart = './data/office_home/Clipart'
    DIR_Product = './data/office_home/Product'
    DIR_REALWORLD = './data/office_home/Real World'

    path = f'data/officehome-{domain}.pkl'
    #trainpath= 'data/officehome-train.pkl'
    #testpath= 'data/officehome-test.pkl'

    officehome_convertor= {'Art':DIR_Art, 'Clipart':DIR_Clipart,'Product':DIR_Product,'Real World':DIR_REALWORLD}
    officehome_transforms_train= transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]) #224,224
    if not os.path.exists(path):
        
        dataset= ImageFolder(officehome_convertor[domain], transform=officehome_transforms_train)
        test_size = len(dataset)

        #Test Set
        test_loader = torch.utils.data.DataLoader(dataset,batch_size=test_size)
        x, y= next(iter(test_loader))
        x= torch.tensor(x)
        y = torch.tensor(y)
        with open(path, 'wb') as f:
            pickle.dump([x, y], f)

    with open(path, 'rb') as f:
        x, y = pickle.load(f)
        if channels == 1:
            x = x[:,0:1,:,:]
    
    if ntr is not None:
        x, y = x[0:ntr], y[0:ntr]
    
    # Without Data Augmentation
    if (translate is None) and (autoaug is None):
        dataset = TensorDataset(x, y)
        return dataset

if __name__=='__main__':
    dataset = load_mnist(split='train')
    print('mnist train', len(dataset))
    dataset = load_mnist('test')
    print('mnist test', len(dataset))
    dataset = load_mnist_m('test')
    print('mnsit_m test', len(dataset))
    dataset = load_svhn(split='test')
    print('svhn', len(dataset))
    dataset = load_usps(split='test')
    print('usps', len(dataset))
    dataset = load_syndigit(split='test')
    print('syndigit', len(dataset))
    dataset= load_stl10(split='test')
    print('stl10', len(dataset))
    dataset= load_cifar10c(split='test')
    print('cifar10-c', len(dataset))
    dataset= load_pacs(split='test')
    print('pacs', len(dataset))
