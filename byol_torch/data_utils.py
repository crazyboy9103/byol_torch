from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader
from torch import nn
import random
import multiprocessing as mp

class DataManager():
  def __init__(self, args):
    norm_stats = {
      "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    }

    train_tranform = BYOLTransform(norm_stats[args.dataset])
    test_transform = NoTransform(norm_stats[args.dataset])

    dataset = datasets.__dict__[args.dataset]
    data_path = f"./data/{args.dataset}"

    self.train_set = dataset(data_path, train=True,  transform=train_tranform, download=True)
    self.test_set  = dataset(data_path, train=False, transform=test_transform, download=True)
    
    self.batch_size = args.batch_size

  def get_loader(self, type):
    if type == "train":
      return DataLoader(
        self.train_set, 
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = mp.cpu_count() // 3,
        pin_memory = True
      )
    
    return DataLoader(
      self.test_set, 
      batch_size = self.batch_size,
      shuffle = True,
      num_workers = mp.cpu_count() // 3,
      pin_memory = True
    )


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
        
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class BYOLTransform(object):
  def __init__(self, norm_stats):
    self.t = T.Compose([
      T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC), 
      T.RandomHorizontalFlip(),
      RandomApply(T.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
      T.GaussianBlur(kernel_size=3),
      T.ToTensor(),
      T.Normalize(*norm_stats),
    ])

    self.t_prime = T.Compose([
      T.RandomResizedCrop(32, interpolation=T.InterpolationMode.BICUBIC), 
      T.RandomHorizontalFlip(),
      RandomApply(T.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
      RandomApply(T.GaussianBlur(kernel_size=3), p=0.1),
      T.RandomSolarize(0.5, 0.2),          
      T.ToTensor(),
      T.Normalize(*norm_stats),
    ])
    
    self.no = NoTransform(norm_stats)
    
  def __call__(self, x):
    return self.no(x), self.t(x), self.t_prime(x)

class NoTransform(object):
  def __init__(self, norm_stats):
      self.base = T.Compose([
          T.ToTensor(), 
          T.Normalize(*norm_stats),
      ])
  
  def __call__(self, x):
      return self.base(x)
