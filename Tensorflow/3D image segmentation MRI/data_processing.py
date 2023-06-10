import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2

HOUNSFIELD_MIN = -1000
HOUNSFIELD_MAX = 2000
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

dataInputPath = 'data/volumes/'
imagePathInput = os.path.join(dataInputPath, 'img/')
makePathInput = os.path.join(dataInputPath, 'mask/')

dataInputPath = 'data/slices/'
imagePathInput = os.path.join(dataInputPath, 'img/')
maskPathInput = os.path.join(dataInputPath, 'mask/')

imgPath = os.path.join(imagePathInput, 'tooth1.nii')
img = nib.load(imgPath).get_fdata()

#imgSlice = img[40,:,:]

maskPath = os.path.join(maskPathInput, 'tooth1.nii')
mask = nib.load(maskPath).get_fdata()

SLICE_X = True
SLICE_Y = True
SLICE_Z = False

SLICE_DECIMATE_IDENTIFIER = 3

def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE


nImg = normalizeImageIntensityRange(img)

def readImageVolume(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        img

#readImageVolume(imgPath, normalize=True)
#readImageVolume(maskPath, normalize=True)

def saveSlice(img, fname, path):
    img =np.uint8(img*255)
    fout = os.path.join(path, f'{fname}.png')
    print(f'[+] Slice saved: {fout}')


imageSliceOutput = ''
maskSliceOutput  = ''

saveSlice(nImg[20,:,:], 'test', imageSliceOutput)
saveSlice(mask[20,:,:], 'test', maskSliceOutput)


def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    cnt = 0
    if SLICE_X:
        cnt += dimx
        for i in range(dimx):
            saveSlice(vol[i,:,:], fname+f' -slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)

    if SLICE_Y:
        cnt += dimy
        for i in range(dimy):
            saveSlice(vol[i,:,:], fname+f' -slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)

    if SLICE_Z:
        cnt += dimx
        for i in range(dimx):
            saveSlice(vol[i,:,:], fname+f' -slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)


for index, filename in enumerate(sorted(glob.iglob(imagePathInput+'*.nii'))):
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 'tooth'+str(index), imageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created\n ')


