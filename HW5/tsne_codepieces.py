
import PIL
from torchvision import models
from torchvision import transforms, utils
from torchvision import datasets, models, transforms

import os,sys,math
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
from sklearn.manifold import TSNE

import torch.nn as nn

import numpy as np

import torch

import sklearn.manifold

from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox


model_urls = {
    'densenet121': None,
    'resnet18': None
}




class dataset_folderdogcat(Dataset):
  def __init__(self, root_dir, trvaltest, perclasslimit, transform=None, offset=0):

    """
    Args:

        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    self.root_dir = root_dir

    self.transform = transform
    self.imgfilenames=[]
    self.labels=[]

    limit=perclasslimit

    for root, directories, filenames in os.walk( os.path.join(self.root_dir,'Dog')):
      ct=0
      for filename in filenames:
        ct+=1
        if ct<offset:
          continue
        if ct+2 >=offset+limit:
          break 
        v=os.path.join(root,filename)
        if os.path.isfile(v):
          #print 
          #print(filename)
          self.imgfilenames.append(os.path.join('Dog',filename))
          self.labels.append(0)


    for root, directories, filenames in os.walk( os.path.join(self.root_dir,'Cat')):
      ct=0
      for filename in filenames:
        ct+=1
        if ct+2 >=limit:
          break 
        v=os.path.join(root,filename)
        if os.path.isfile(v):
          #print 
          #print(filename)
          self.imgfilenames.append(os.path.join('Cat',filename))
          self.labels.append(1)

    print(np.mean(np.asarray(self.labels)))

  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):
    image = PIL.Image.open( os.path.join(self.root_dir,self.imgfilenames[idx]))


    #label=self.labels[idx,:].astype(np.float32)


    if self.transform:
      image = self.transform(image)

    # if you do five crop, thne you MUST change this part, as outputs are now 4 d tensors!!!
    if image.size()[0]==1:
      image=image.repeat([3,1,1])
    if image.size()[0]==4:
      image=image[0:3,:,:]

    #print(self.imgfilenames[idx])

    sample = {'image': image,  'filename': self.imgfilenames[idx], 'label': self.labels[idx]}

    return sample



def fcomp():

  pt= 'PetImages/' 
  savedir='dogcatfeats_0'



  if not os.path.isdir(savedir):
    os.makedirs(savedir)

  device=torch.device('cpu')

 
  BATCH_SIZE=32
  imgsize=64

  data_transform = transforms.Compose([
               transforms.Scale(imgsize),
               transforms.CenterCrop(imgsize),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
              ])

  dataset = dataset_folderdogcat(root_dir=pt,trvaltest=-1, perclasslimit = 200, transform=data_transform , offset=5000)

  print('len(dataset)',len(dataset))

  dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

  ############################################
  ############################################
  ############################################
  model=models.resnet18(pretrained=True).to(device)
  #your pretrained model.to(device)
  model.eval()
  # print(model)


  for k,dic in enumerate(dataset_loader):
    print(k)

    fnames=dic['filename']
    #print(fnames)

    imgs=dic['image']
    with torch.no_grad():

      ############################################
      ############################################
      ############################################
      # modify your model so that it returns the prediction p and the feature after adaptiveaveragepooling

      # p,fts=model.forward3('avgpool',imgs.to(device))
      res18_conv = nn.Sequential(*list(model.children())[:-2])
      fts=res18_conv(imgs.to(device))

      npfts=fts.to('cpu').numpy()

    print(npfts.shape)
    for i,fn in enumerate(fnames):
      savename=os.path.join(savedir, fn+'_ft.npy')
      curdir=  os.path.dirname(savename)
      if not os.path.isdir(curdir):
        os.makedirs(curdir)
      print(savename)
 
      np.save(savename,npfts[i,:])


def visfeats():


  pt= 'PetImages/' 
  savedir='dogcatfeats_0'


  imgsize=64

  data_transform = transforms.Compose([
               transforms.Scale(imgsize),
               transforms.CenterCrop(imgsize),
               transforms.ToTensor(),
               #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
              ])

  dataset = dataset_folderdogcat(root_dir=pt,trvaltest=-1, perclasslimit = 200, transform=data_transform , offset=5000)

  #laod features and images
  imgs=[]
  fts=None
  for i,dic in enumerate(dataset):

    fn=dic['filename']
    savename=os.path.join(savedir, fn+'_ft.npy')
    ft=np.load(savename)
    #ft=np.mean(np.mean(ft,axis=2),axis=1) #deep inception layers

    if fts is None:
      fts=ft[np.newaxis,:]
    else:
      fts=np.concatenate( (fts,  ft[np.newaxis,:]  ), axis=0  )

    imgs.append(   np.moveaxis(dic['image'].numpy(),0,2))
    # print (imgs[0].shape)

  ############################################
  ############################################
  ############################################
  ##### run scikit learn tsne and visualization here
  # print(fts.shape)
  fts = np.reshape(fts, (fts.shape[0], -1))
  print(fts.shape)
  X_embedded = TSNE(n_components=2).fit_transform(fts)
  plot_embedding(X_embedded, imgs)
  #plot the embedding using plot_embedding


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors

#X as outputted by sklearn.manifold.TSNE
#images a list of numpy arrays
# images[i].shape=(h,w,c)
def plot_embedding(X, images, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str('0'),
                 color=plt.cm.Set1(0 / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            #print('images[i].shape', images[i].shape)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(  images[i],zoom=0.6  ),
                xy=X[i] )
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])

    plt.savefig('{}.png'.format(i))
    if title is not None:
        plt.title(title)





if __name__=='__main__':
  fcomp()
  visfeats()
  # catdogtrain()

  #fcomp2()
  #visfeats2()

