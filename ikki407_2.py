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
from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array
import cv2
from scipy import ndimage

#DEPLOY_MODEL = '/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/deploy_caffe_bowl_1.prototxt'
DEPLOY_MODEL = '/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/caffe_bowl_2.prototxt'
MODEL = '/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/bowl_train_iter_305000.caffemodel'

#net = caffe.Net(DEPLOY_MODEL,MODEL,image_dims=(96, 96))
#net.crop_dims
#net.set_phase_test()# test phaseで定義されたcaffe netを使用する
#net.set_raw_scale('data', 255)  # data layerに入力として与えられる画像の輝度上限を指定
MEAN_FILE = "/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/meanfile.binaryproto"
blob = caffe_pb2.BlobProto()
blob.ParseFromString(open(MEAN_FILE, "rb").read())
mean = blobproto_to_array(blob).reshape(1,96,96)
#net.set_mean('data', mean)
#scores = net.predict([caffe.io.load_image('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/test/1.jpg', color = False, )], oversample=False)

import leveldb
db = leveldb.LevelDB('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/test_leveldb/')
test_data_raw =  [v for v in db.RangeIter()]
def toDatum(x):
    return caffe_pb2.Datum.FromString(x[1])
test_datum_set = map(toDatum, test_data_raw)

# datumからデータ(イメージ)とラベルを抽出
# イメージデータの次元(shape)は(N, C(色数), Height ,Width)なので、(N, Height, Width, C)に変換しておく
test_data = np.rollaxis(np.array(map(caffe.io.datum_to_array, test_datum_set)),1,4)
#input_data = [d - mean.transpose(1, 2, 0) for d in test_data]
test_label = np.array(map(lambda x: x.label, test_datum_set))

net = caffe.Net('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/deploy_caffe_bowl_3.prototxt',MODEL)
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
    #print("start_time:{0}".format(start))


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


    #net.blobs['data'] = input_

    # Classify
    caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],dtype=np.float32)
    input2_ = [d - mean.transpose(1, 2, 0) for d in input_]
    net.set_phase_test()
    for ix, in_ in enumerate(input2_):
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


SampleSubmit = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/sampleSubmission.csv')


Class = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/Class_aug.txt',header=None,sep=' ')
Class = Class.sort(columns=[0])

#prediction = DataFrame(scores)
prediction = scores[list(Class[1])]
prediction.columns = Class[0]
prediction = prediction[SampleSubmit.columns[1:]]
#SampleSubmit = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/sampleSubmission.csv')

image_list = DataFrame([item[0][9:].replace('_96_96', '') for item in test_data_raw])#予測値のimage name
image_list.index = prediction.index
Prediction = pd.concat([image_list,prediction],axis=1)
Prediction.columns = SampleSubmit.columns

#DataFrame(SampleSubmit['image'],).join(Prediction, on=u'image',how='left')
#Prediction = pd.concat([DataFrame(SampleSubmit['image']),Prediction],join='outer',keys='image',ignore_index=True)

Prediction = pd.merge(DataFrame(SampleSubmit['image']),Prediction,how='outer')

if len(SampleSubmit) == len(Prediction.dropna()):
    if list(Prediction.columns) ==(list(SampleSubmit.columns)):
        print '全データ予測完了,提出可能形式'

Prediction.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/submit5.csv',index=None)


blobs = net.blobs
[(k,v.data[0].shape) for k,v in blobs.items()]
#画像を見る

def plot_images(images, tile_shape):
    assert images.shape[0] <= (tile_shape[0]* tile_shape[1])
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols =tile_shape)
    print images.shape[0]
    for i in range(images.shape[0]):
        grd=grid[i]
        grd.imshow(images[i].reshape(100,100))
plot_images(test_data[100:104],tile_shape=(2,2))


"""
DeepNetsの図
"""
import pydot
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from caffe.draw import get_pydot_graph
from IPython.display import display, Image 

net_param = caffe_pb2.NetParameter()
f = open('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/caffe_bowl_3.prototxt')
text_format.Merge(f.read(), net_param)
display(Image(get_pydot_graph(net_param).create_png()))


"""
train,val,testデータのtxtを作成
ここで、trainとvalの分割を行っている
X:ファイル名
Y:Label
アフィン変換する

"""
'''
import cv2
import numpy as np
#画像読み込み
img = cv2.imread('/Users/IkkiTanaka/Desktop/スクリーンショット 2015-01-19 15.22.31.png')
rows,cols,ch = img.shape
#アフィン変換
pts1 = np.float32([[0,0],[0,rows],[cols,0]])
pts2 = np.float32([[0,0],[0,96],[96,0]])

M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))

#plt.subplot(121),plt.imshow(img),plt.title('Input')
#plt.subplot(122),plt.imshow(dst),plt.title('Output')
#余計なとこ切り抜き
dstf = dst[40:96, 0:96]
#dst2 = cv2.resize(img,(96,96))
#plt.subplot(121),plt.imshow(dst),plt.title('Affine')
#plt.subplot(122),plt.imshow(dst2),plt.title('resize')
#plt.show()
#保存
cv2.imwrite('大森.jpg', dst)

#Rotation
M = cv2.getRotationMatrix2D((96/2,96/2),90,1)
rotated = cv2.warpAffine(dst,M,(96,96))
plt.subplot(121),plt.imshow(dst)
plt.subplot(122),plt.imshow(rotated)
plt.show()

#左右反転
dstf = cv2.flip(dst,1,dstf)
plt.subplot(121),plt.imshow(dst),plt.title('original')
plt.subplot(122),plt.imshow(dstf),plt.title('mirror')

plt.show()

#画像読み込み
img = cv2.imread('/Users/IkkiTanaka/Desktop/スクリーンショット 2015-01-19 15.22.31.png')
rows,cols,ch = img.shape
#アフィン変換
pts1 = np.float32([[0,0],[0,rows],[cols,0]])
pts2 = np.float32([[0,0],[0,96],[96,0]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
dst = dst[0:96, 0:96]
#保存
cv2.imwrite('大森.jpg', dst)
#Rotation
M = cv2.getRotationMatrix2D((96/2,96/2),90,1)
rotated = cv2.warpAffine(dst,M,(96,96))
#plt.subplot(121),plt.imshow(dst)
#plt.subplot(122),plt.imshow(rotated)
#plt.show()

#左右反転
dstf = cv2.flip(dst,1)
plt.subplot(121),plt.imshow(dst),plt.title('original')
plt.subplot(122),plt.imshow(dstf),plt.title('mirror')

plt.show()

#アス比考慮した変形

img = Image.open('/Users/IkkiTanaka/Desktop/スクリーンショット 2015-01-19 15.22.31.png')
simg = img.copy()
simg.thumbnail((96, 96), Image.ANTIALIAS)
white_img = Image.new('RGBA', (96, 96),'white')
if simg.size[0]<=simg.size[1]:
    white_img.paste(simg,((96-simg.size[0])/2,0))
else:
    white_img.paste(simg,(0,(96-simg.size[1])/2))
plt.subplot(121),plt.imshow(img),plt.title('original')
plt.subplot(122),plt.imshow(white_img),plt.title('mirror')

'''


"""
OpenCVで画像処理
"""

#caffe用にtrainデータをtxtに落としこむ
directory_names = list(set(glob.glob(os.path.join("/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/","original_train", "*"))\
 ).difference(set(glob.glob(os.path.join("/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/","original_train","*.*")))))

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
            #画像読み込み
            img = cv2.imread(Original_image)
            rows,cols,ch = img.shape#imageのサイズ
            
            #アフィン変換(96×96) Rotation=0
            pts1 = np.float32([[0,0],[0,rows],[cols,0]])
            pts2 = np.float32([[0,0],[0,96],[96,0]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(96,96))
            dst = dst[0:96, 0:96]#切り抜き
            new_fileName = fileName[:-4] + '_72_72'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dst)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #左右反転 rot=0
            dstf = cv2.flip(dst,1)
            new_fileName = fileName[:-4] + '_72_72_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            
            #Rotation=90
            #M = cv2.getRotationMatrix2D((96/2,96/2),90,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,90)
            new_fileName = fileName[:-4] + '_72_72_90'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #左右反転 rot=90
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_90_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            
            #Rotation=180
            #M = cv2.getRotationMatrix2D((96/2,96/2),180,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,180)
            new_fileName = fileName[:-4] + '_72_72_180'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #左右反転 rot=180
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_180_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #Rotation=270
            #M = cv2.getRotationMatrix2D((96/2,96/2),270,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,270)            
            new_fileName = fileName[:-4] + '_72_72_270'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            #左右反転 rot=270
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_270_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)




            #100_100にして左上を96_96にcropping Rotation=0
            #dst
            #アフィン変換(96×96) Rotation=0
            pts1 = np.float32([[0,0],[0,rows],[cols,0]])
            pts2 = np.float32([[0,0],[0,100],[100,0]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(100,100))
            dst = dst[0:96, 0:96]#切り抜き
            new_fileName = fileName[:-4] + '_72_72_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dst)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #distf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(dst,1)
            new_fileName = fileName[:-4] + '_72_72_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=90
            #M = cv2.getRotationMatrix2D((96/2.0,96/2.0),90,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,90)              
            new_fileName = fileName[:-4] + '_72_72_90_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)          
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_90_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=180
            #M = cv2.getRotationMatrix2D((96/2,96/2),180,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,180)  
            new_fileName = fileName[:-4] + '_72_72_180_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_180_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=270
            #M = cv2.getRotationMatrix2D((96/2,96/2),270,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,270)              
            new_fileName = fileName[:-4] + '_72_72_270_croplt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_270_croplt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)


            #100_100にして左下を96_96にcropping Rotation=0
            #dst
            #アフィン変換(96×96) Rotation=0
            pts1 = np.float32([[0,0],[0,rows],[cols,0]])
            pts2 = np.float32([[0,0],[0,100],[100,0]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(100,100))
            dst = dst[4:100, 0:96]#切り抜き
            new_fileName = fileName[:-4] + '_72_72_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dst)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #distf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(dst,1)
            new_fileName = fileName[:-4] + '_72_72_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=90
            #M = cv2.getRotationMatrix2D((96/2,96/2),90,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,90)  
            new_fileName = fileName[:-4] + '_72_72_90_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)            
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_90_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=180
            #M = cv2.getRotationMatrix2D((96/2,96/2),180,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,180)
            new_fileName = fileName[:-4] + '_72_72_180_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_180_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=270
            #M = cv2.getRotationMatrix2D((96/2,96/2),270,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,270)
            new_fileName = fileName[:-4] + '_72_72_270_croplb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_270_croplb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)


            #100_100にして右上を96_96にcropping Rotation=0
            #dst
            #アフィン変換(96×96) Rotation=0
            pts1 = np.float32([[0,0],[0,rows],[cols,0]])
            pts2 = np.float32([[0,0],[0,100],[100,0]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(100,100))
            dst = dst[0:96, 4:100]#切り抜き
            new_fileName = fileName[:-4] + '_72_72_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dst)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #distf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(dst,1)
            new_fileName = fileName[:-4] + '_72_72_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=90
            #M = cv2.getRotationMatrix2D((96/2,96/2),90,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,90)
            new_fileName = fileName[:-4] + '_72_72_90_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)            
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_90_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=180
            #M = cv2.getRotationMatrix2D((96/2,96/2),180,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,180)
            new_fileName = fileName[:-4] + '_72_72_180_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_180_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=270
            #M = cv2.getRotationMatrix2D((96/2,96/2),270,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,270)
            new_fileName = fileName[:-4] + '_72_72_270_croprt'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_270_croprt_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            
            #100_100にして右下を96_96にcropping Rotation=0
            #dst
            #アフィン変換(96×96) Rotation=0
            pts1 = np.float32([[0,0],[0,rows],[cols,0]])
            pts2 = np.float32([[0,0],[0,100],[100,0]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(100,100))
            dst = dst[4:100, 4:100]#切り抜き
            new_fileName = fileName[:-4] + '_72_72_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dst)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #distf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(dst,1)
            new_fileName = fileName[:-4] + '_72_72_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=90
            #M = cv2.getRotationMatrix2D((96/2,96/2),90,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,90)
            new_fileName = fileName[:-4] + '_72_72_90_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)            
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_90_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=180
            #M = cv2.getRotationMatrix2D((96/2,96/2),180,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,180)
            new_fileName = fileName[:-4] + '_72_72_180_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_180_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #rotated
            #Rotation=270
            #M = cv2.getRotationMatrix2D((96/2,96/2),270,1)
            #rotated = cv2.warpAffine(dst,M,(96,96))
            rotated = ndimage.rotate(dst,270)
            new_fileName = fileName[:-4] + '_72_72_270_croprb'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, rotated)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)
            #dstf
            #左右反転 rot=0 mirror 96_96
            dstf = cv2.flip(rotated,1)
            new_fileName = fileName[:-4] + '_72_72_270_croprb_mirror'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dstf)
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

train_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/train_aug.txt',index=None,header=False,sep=' ')
val_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/val_aug.txt',index=None,header=False,sep=' ')
Class_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/Class_aug.txt',index=None,header=False,sep=' ')


#caffe用にtestデータをtxtに落としこむ
i = 0    
label = 0
# List of string of class names
namesClasses = list()
X_pred = []
Y_pred = []
directory_names_pred = ["/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/original_test"]
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
                        
            #画像読み込み
            img = cv2.imread(Original_image)
            rows,cols,ch = img.shape#imageのサイズ
            
            #アフィン変換(100×100) Rotation=0
            pts1 = np.float32([[0,0],[0,rows],[cols,0]])
            pts2 = np.float32([[0,0],[0,100],[100,0]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(100,100))
            dst = dst[0:100, 0:100]#切り抜き
            new_fileName = fileName[:-4] + '_96_96'+'.jpg'
            new_Dir = '/'.join(['/'.join(folder.split(os.sep)),new_fileName])
            #保存
            cv2.imwrite(new_Dir, dst)
            X.append('/'.join([folder.split(os.sep)[-1],new_fileName]))
            Y.append(label)

            i+=1
            print i

train_pred_txt = DataFrame([X_pred,Y_pred]).T

train_pred_txt.to_csv('/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/test_aug.txt',index=None,header=False,sep=' ')


"""
terminalでの操作 #-gray=1
"""
#train
/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -gray=1 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/original_train/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/train_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/train_leveldb 

#/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -shuffle=1 -resize_height 48 -resize_width 48 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/train_leveldb_2 

#validation
/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -gray=1 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/original_train/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/val_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/val_leveldb 

#平均画像作成
/Users/IkkiTanaka/caffe/build/tools/compute_image_mean /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/train_leveldb /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/meanfile.binaryproto leveldb

#test
/Users/IkkiTanaka/caffe/build/tools/convert_imageset -backend="leveldb" -gray=1 /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/original_test/ /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/test_aug.txt /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/test_leveldb 

#学習
/Users/IkkiTanaka/caffe/.build_release/tools/caffe train -solver /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/train_val_solver.prototxt

#学習再開
/Users/IkkiTanaka/caffe/.build_release/tools/caffe train -solver /Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/train_val_solver.prototxt --snapshot=/Users/IkkiTanaka/Documents/kaggle/National_Data_Science_Bowl/ikki2/bowl_train_iter_305000.solverstate



