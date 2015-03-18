#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
# make graphics inline
#%matplotlib inline
from pandas import DataFrame
import caffe
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
"""
train,val,testデータのtxtを作成
ここで、trainとvalの分割を行っている
X:ファイル名
Y:Label
"""
#caffe用にtrainデータをtxtに落としこむ
directory_names = list(set(glob.glob(os.path.join("/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/","original_train", "*"))\
 ).difference(set(glob.glob(os.path.join("/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/","original_train","*.*")))))

i = 0    
label = 0
# List of string of class names
namesClasses = list()
X = []
Y = []
#width_resize = 96
#height_resize = 96
for folder in directory_names:
    # Append the string class name for each class
    currentClass = folder.split(os.sep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            #file名->fileName Path='/'.join(['/'.join(folder.split(os.sep)),fileName])
            Original_image = '/'.join(['/'.join(folder.split(os.sep)),fileName])
            im = Image.open(Original_image)
            width, height = im.size #imageのサイズ
            
            #72_72 rot=0
            image_72_72 = im.resize((96,96), Image.ANTIALIAS)#リサイズ
            new_fileName = fileName[:-4] + '_72_72'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            
            #72_72 rot=0 mirror
            image_72_72_mirror = image_72_72.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #72_72 rot=90
            image_72_72_90 = image_72_72.rotate(90)#回転
            new_fileName = fileName[:-4] + '_72_72_90'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            
            #72_72 rot=90 mirror
            image_72_72_90_mirror = image_72_72_90.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_90_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #72_72 rot=180
            image_72_72_180 = image_72_72.rotate(180)#回転
            new_fileName = fileName[:-4] + '_72_72_180'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            
            #72_72 rot=180 mirror
            image_72_72_180_mirror = image_72_72_180.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_180_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #72_72 rot=270
            image_72_72_270 = image_72_72.rotate(270)#回転
            new_fileName = fileName[:-4] + '_72_72_270'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #72_72 rot=270 mirror
            image_72_72_270_mirror = image_72_72_270.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_270_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #96_96にして左上を72_72にcropping rot=0
            image_96_96 = im.resize((100,100), Image.ANTIALIAS)#リサイズ
            image_72_72_crop = image_96_96.crop( (0,0 , 96, 96))
            new_fileName = fileName[:-4] + '_72_72_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_crop_mirror = image_72_72_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=90
            image_72_72_90_crop  = image_72_72_crop.rotate(90)#回転
            new_fileName = fileName[:-4] + '_72_72_90_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=90 mirror
            image_72_72_90_crop_mirror = image_72_72_90_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_90_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=180
            image_72_72_180_crop = image_72_72_crop.rotate(180)#回転
            new_fileName = fileName[:-4] + '_72_72_180_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_180_crop_mirror = image_72_72_180_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_180_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=270
            image_72_72_270_crop = image_72_72_crop.rotate(270)#回転
            new_fileName = fileName[:-4] + '_72_72_270_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_270_crop_mirror = image_72_72_270_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_270_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)


            #96_96にして左下を72_72にcropping rot=0
            image_96_96 = im.resize((100,100), Image.ANTIALIAS)#リサイズ
            image_72_72_crop = image_96_96.crop((0,4 , 96, 100))
            new_fileName = fileName[:-4] + '_72_72_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_crop_mirror = image_72_72_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=90
            image_72_72_90_crop  = image_72_72_crop.rotate(90)#回転
            new_fileName = fileName[:-4] + '_72_72_90_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=90 mirror
            image_72_72_90_crop_mirror = image_72_72_90_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_90_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=180
            image_72_72_180_crop = image_72_72_crop.rotate(180)#回転
            new_fileName = fileName[:-4] + '_72_72_180_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_180_crop_mirror = image_72_72_180_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_180_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=270
            image_72_72_270_crop = image_72_72_crop.rotate(270)#回転
            new_fileName = fileName[:-4] + '_72_72_270_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_270_crop_mirror = image_72_72_270_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_270_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)


            #96_96にして右上を72_72にcropping rot=0
            image_96_96 = im.resize((100,100), Image.ANTIALIAS)#リサイズ
            image_72_72_crop = image_96_96.crop((4,0 , 100, 96))
            new_fileName = fileName[:-4] + '_72_72_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_crop_mirror = image_72_72_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=90
            image_72_72_90_crop  = image_72_72_crop.rotate(90)#回転
            new_fileName = fileName[:-4] + '_72_72_90_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=90 mirror
            image_72_72_90_crop_mirror = image_72_72_90_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_90_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=180
            image_72_72_180_crop = image_72_72_crop.rotate(180)#回転
            new_fileName = fileName[:-4] + '_72_72_180_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_180_crop_mirror = image_72_72_180_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_180_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=270
            image_72_72_270_crop = image_72_72_crop.rotate(270)#回転
            new_fileName = fileName[:-4] + '_72_72_270_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_270_crop_mirror = image_72_72_270_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_270_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)


            #96_96にして右下を72_72にcropping rot=0
            image_96_96 = im.resize((100,100), Image.ANTIALIAS)#リサイズ
            image_72_72_crop = image_96_96.crop((4,4 , 100, 100))
            new_fileName = fileName[:-4] + '_72_72_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_crop_mirror = image_72_72_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=90
            image_72_72_90_crop  = image_72_72_crop.rotate(90)#回転
            new_fileName = fileName[:-4] + '_72_72_90_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=90 mirror
            image_72_72_90_crop_mirror = image_72_72_90_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_90_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_90_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=180
            image_72_72_180_crop = image_72_72_crop.rotate(180)#回転
            new_fileName = fileName[:-4] + '_72_72_180_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_180_crop_mirror = image_72_72_180_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_180_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_180_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #96_96にして左上を72_72にcropping rot=270
            image_72_72_270_crop = image_72_72_crop.rotate(270)#回転
            new_fileName = fileName[:-4] + '_72_72_270_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #72_72 rot=0 mirror
            image_72_72_270_crop_mirror = image_72_72_270_crop.transpose(Image.FLIP_LEFT_RIGHT)
            new_fileName = fileName[:-4] + '_72_72_270_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_72_72_270_crop_mirror.save(new_Dir)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            i += 1
    print currentClass, label
    label += 1

from sklearn.cross_validation import train_test_split
train_txt = DataFrame([X,Y]).T
Class_txt = DataFrame([namesClasses,range(121)]).T

#train用とvalidation用に分割
#defaultでは3:1
train_txt, val_txt = train_test_split(train_txt,test_size=0.05, random_state=407)

train_txt = DataFrame(train_txt)
val_txt = DataFrame(val_txt)

train_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_aug.txt',index=None,header=False,sep=' ')
val_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/val_aug.txt',index=None,header=False,sep=' ')
Class_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/Class_aug.txt',index=None,header=False,sep=' ')


#caffe用にtestデータをtxtに落としこむ
i = 0    
label = 0
# List of string of class names
namesClasses = list()
X_pred = []
Y_pred = []
directory_names_pred = ["/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/original_test"]
for folder in directory_names_pred:
    # Append the string class name for each class
    currentClass = folder.split(os.sep)[-1]
    namesClasses.append(currentClass)
    for fileNameDir in os.walk(folder):#sortすればアルファベット順になる？   
        for fileName in fileNameDir[2]:
            # Only read in the images
            if fileName[-4:] != ".jpg":
              continue
            Original_image = '/'.join(['/'.join(folder.split(os.sep)),fileName])
            im = Image.open(Original_image)
            width, height = im.size #imageのサイズ
            
            #72_72 rot=0
            image_96_96 = im.resize((100,100), Image.ANTIALIAS)#リサイズ
            new_fileName = fileName[:-4] + '_96_96'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            image_96_96.save(new_Dir)
            X_pred.append(new_fileName)
            Y_pred.append(label)
            i+=1
            print i

train_pred_txt = DataFrame([X_pred,Y_pred]).T

train_pred_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/test_aug.txt',index=None,header=False,sep=' ')


"""
terminalでの操作 #-gray=1
"""
#train
/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -gray=1 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/original_train/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_leveldb 

#/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -shuffle=1 -resize_height 48 -resize_width 48 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_leveldb_2 

#validation
/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -gray=1 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/original_train/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/val_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/val_leveldb 

#平均画像作成
/Users/IkkiTanaka/caffe/build/tools/compute_image_mean /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_leveldb /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/meanfile.binaryproto leveldb

#test
/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -gray=1 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/original_test/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/test_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/test_leveldb 

#学習
/Users/IkkiTanaka/caffe/.build_release/tools/caffe train -solver /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_val_solver.prototxt

#学習再開
/Users/IkkiTanaka/caffe/.build_release/tools/caffe train -solver /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_val_solver.prototxt --snapshot=/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/bowl_train_iter_305000.solverstate



"""
予測
"""
from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array

#DEPLOY_MODEL = '/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/deploy_caffe_bowl_1.prototxt'
DEPLOY_MODEL = '/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/caffe_bowl_1.prototxt'
MODEL = '/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/bowl_train_iter_170000.caffemodel'

#net = caffe.Net(DEPLOY_MODEL,MODEL,image_dims=(96, 96))
#net.crop_dims
#net.set_phase_test()# test phaseで定義されたcaffe netを使用する
#net.set_raw_scale('data', 255)  # data layerに入力として与えられる画像の輝度上限を指定
MEAN_FILE = "/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/meanfile.binaryproto"
blob = caffe_pb2.BlobProto()
blob.ParseFromString(open(MEAN_FILE, "rb").read())
mean = blobproto_to_array(blob).reshape(1,96,96)
#net.set_mean('data', mean)
#scores = net.predict([caffe.io.load_image('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/test/1.jpg', color = False, )], oversample=False)

import leveldb
db = leveldb.LevelDB('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/test_leveldb/')
test_data_raw =  [v for v in db.RangeIter()]
def toDatum(x):
    return caffe_pb2.Datum.FromString(x[1])
test_datum_set = map(toDatum, test_data_raw)

# datumからデータ(イメージ)とラベルを抽出
# イメージデータの次元(shape)は(N, C(色数), Height ,Width)なので、(N, Height, Width, C)に変換しておく
test_data = np.rollaxis(np.array(map(caffe.io.datum_to_array, test_datum_set)),1,4)
#input_data = [d - mean.transpose(1, 2, 0) for d in test_data]
test_label = np.array(map(lambda x: x.label, test_datum_set))

def plot_images(images, tile_shape):
    assert images.shape[0] <= (tile_shape[0]* tile_shape[1])
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols =tile_shape)
    print images.shape[0]
    for i in range(images.shape[0]):
        grd=grid[i]
        grd.imshow(images[i].reshape(96,96))
plot_images(test_data[100:104],tile_shape=(2,2))
"""
予測
"""

#scores = net.predict(test_data, oversample=True)

#inputs = test_data
# Scale to standardize input dimensions.
#input_ = np.zeros((len(inputs),net.image_dims[0], net.image_dims[1], inputs[0].shape[2]),dtype=np.float32)
#for ix, in_ in enumerate(inputs):
#    input_[ix] = caffe.io.resize_image(in_, net.image_dims)

net = caffe.Net('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/deploy_caffe_bowl_2.prototxt',MODEL)
net.crop_dims=(96,96)
net.set_phase_test()
#prob = net.forward_all(blobs=['prob'])
#net.set_mean('data', mean)

#net.forward()
#prob = net.blobs['prob'].data # or whatever you want


import time
oversample=True
scores = DataFrame(np.zeros((len(test_data),121)))
for aaa in range(0,len(test_data),10000):
    start = time.time()
    print("start_time:{0}".format(start))


    if aaa > len(test_data)-10000:
        input_ = test_data[aaa:len(test_data)]
    else:
        input_ = test_data[aaa:(aaa+10000)]

    #input0_ = np.zeros((len(input_),net.image_dims[0], net.image_dims[1], input_[0].shape[2]),dtype=np.float32)
    #for ix, in_ in enumerate(input_):
    #    input0_[ix] = caffe.io.resize_image(in_, net.image_dims)

    #ここで、input_をtest_dataを用いて標準化しないで予測する
    if oversample:
        # Generate center, corner, and mirrored crops.
        input_ = caffe.io.oversample(input_, net.crop_dims)
    else:
    # Take center crop.
        center = np.array(net.image_dims) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([-net.crop_dims / 2.0,net.crop_dims / 2.0])
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]


    net.blobs['data'] = input_

    # Classify
    caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],dtype=np.float32)
    #in_ = [d - mean.transpose(1, 2, 0) for d in input_]
    net.set_phase_test()
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = net.preprocess('data', in_)
    
    #net.preprocess(net.inputs[0], in_)
    #input_data = [d - mean.transpose(1, 2, 0) for d in input_]
    net.set_phase_test()
    out = net.forward_all(**{'data': caffe_in})
    predictions = out['prob'].squeeze(axis=(2,3))

    # For oversampling, average predictions across crops.
    if oversample:
        predictions = predictions.reshape((len(predictions) / 10, 10, -1))
        predictions = predictions.mean(1)

    score = DataFrame(predictions)
    if aaa > len(test_data)-10000:
        score.index = scores[aaa:len(test_data)].index
        scores[aaa:len(test_data)] = score
    else:
        score.index = scores[aaa:(aaa+10000)].index
        scores[aaa:(aaa+10000)] = score
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time))




Class = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/Class_aug.txt',header=None,sep=' ')
Class = Class.sort(columns=[0])

#prediction = DataFrame(scores)
prediction = scores[list(Class[1])]

SampleSubmit = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/sampleSubmission.csv')

image_list = DataFrame([item[0][9:].replace('_96_96', '') for item in test_data_raw])#予測値のimage name
image_list.index = prediction.index
Prediction = pd.concat([image_list,prediction],axis=1)
Prediction.columns = SampleSubmit.columns

#DataFrame(SampleSubmit['image'],).join(Prediction, on=u'image',how='left')
#Prediction = pd.concat([DataFrame(SampleSubmit['image']),Prediction],join='outer',keys='image',ignore_index=True)

Prediction = pd.merge(DataFrame(SampleSubmit['image']),Prediction,how='outer')

if len(SampleSubmit) == len(Prediction.dropna()):
    print '全データ予測完了,提出可能形式'

Prediction.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/submit3.csv',index=None)


blobs = net.blobs
[(k,v.data[0].shape) for k,v in blobs.items()]
