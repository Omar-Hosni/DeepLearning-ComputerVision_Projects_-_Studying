from tensorflow.keras.models import load_model
import nibabel as nib
from niwidgets import NiftiWidget
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

SLICE_X = True
SLICE_Y = True
SLICE_Z = False
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 80
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return(img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

targetName=  'tooth5'
targetImagePath = f'data/volumes/img/{targetName}.nii'
targetMaskPath = f'data/volumes/mask/{targetName}.nii'

imgTargetNii = nib.load(targetImagePath)
imgMaskNii = nib.load(targetMaskPath)

imgTarget = normalizeImageIntensityRange(imgTargetNii.get_fdata())
imgMask = imgMaskNii.get_fdata()

model = load_model('UNET-ToothSegmentation_40_50.h5')

def scaleImg(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)


sliceIndex = 24

plt.figure(figsize=(15,15))
imgSlice = imgTarget[sliceIndex,:,:]
imgSliceScaled = scaleImg(imgSlice, IMAGE_HEIGHT, IMAGE_WIDTH)
imgDimX, imgDimY =imgSlice.shape
plt.show()

plt.figure(figsize=(15,15))
maskSlice = imgMask[sliceIndex,:,:]
maskSliceScaled = scaleImg(maskSlice, IMAGE_HEIGHT, IMAGE_WIDTH)
plt.subplot(1,2,2)
plt.imshow(maskSliceScaled, cmap='gray')
plt.show()

#predict with unit model
plt.figure(figsize=(15,15))
imageInput = imgSliceScaled[np.newaxis, :, :,np.newaxis]
maskPredict = model.predict(imageInput)
plt.subplot(1,2,2)
plt.imshow(maskPredict, cmap='gray')

def predictVolume(inImg, toBin=True):
    (xMax, yMax, zMax) = inImg.shape
    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))

    cnt = 0.0
    if SLICE_X:
        cnt += 1.0
        for i in range(xMax):
            img = scaleImg(inImg[i,:,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgX[i,:,:] = scaleImg(tmp,yMax,zMax)

    if SLICE_Y:
        cnt += 1.0
        for i in range(xMax):
            img = scaleImg(inImg[i, :, :], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis, :, :, np.newaxis]
            tmp = model.predict(img)[0, :, :, 0]
            outImgX[:, i, :] = scaleImg(tmp, xMax, zMax)

    if SLICE_Z:
        cnt += 1.0
        for i in range(xMax):
            img = scaleImg(inImg[i, :, :], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis, :, :, np.newaxis]
            tmp = model.predict(img)[0, :, :, 0]
            outImgX[:, :, i] = scaleImg(tmp, xMax, yMax)

    outImg = (outImgX +outImgY + outImgZ) / cnt

    if(toBin):
        outImg[outImg > 0.5] = 1.0
        outImg[outImg <= 0.5] = 0.0
    return outImg


predImg = predictVolume(imgTarget)
my_widget = NiftiWidget(imgTargetNii)
my_widget.nifti_plotter(colormap='gray')

my_widget = NiftiWidget()

#Convert binary image to mesh
from skimage.measure import _marching_cubes_lewiner
import meshplot as mp
from stl import mesh

vertices, faces, _, _ = _marching_cubes_lewiner
mp.plot(vertices, faces)

def dataToMesh(vert, faces):
    mm = mesh.Mesh
    for i, f in enumerate(faces):
        for j in range(3):
            mm.vectors[i][j] = vert[f[j],:]
    return mm

mm = dataToMesh(vertices, faces)
mm.save('tooth-segmented.stl')