#import dicom2nifti

in_path = 'D:\projects\Tumor Detection\Task03 Liver smaller\labels'
out_path = 'D:\projects\Tumor Detection\Task03 Liver smaller\labels'

#from glob2 import glob
#import shutil
#import os
#import nibabel as nib
import numpy as np

sens = 0.9
preve =  0.2
speci = 0.8

ppv = (sens * preve) / ((sens * preve) + (1-speci) * (1-preve))
print(ppv)

#acc = P(correct|disease)P(disease) + P(correct|normal)P(normal)
#acc = sensivity*P(disease) + specifity*P(normal)
#acc = sensitivity * prevelance + specifity * (1-prevelence)
acc = sens * preve + speci * (1-preve)
print(acc)



'''
#CREATE GROUPS OF 65 SLICES

patient_list = glob(in_path+"/*")
print(patient_list[0]) #first item of list, first patient

for patient in glob(in_path + '/*'):
    patient_name = os.path.basename(os.path.normpath(patient))
    number_folders = int(glob(patient+"/*")/64)

    for i in range(number_folders):
        output_path_name = os.path.join(out_path, patient_name + '_' + i)
        os.mkdir(output_path_name)

        for i, file in enumerate(glob(patient('/*'))):
            if i == 64+1:
                break
            shutil.move(file, output_path_name)


#CREATE THE DICOM FILES INTO NIFTIES
in_path_images = 'D:\projects\Tumor Detection\Task03 Liver smaller\dicom_groups\images'
out_path_images = 'D:\projects\Tumor Detection\Task03 Liver smaller\\nifti_files\images'

in_path_labels = 'D:\projects\Tumor Detection\Task03 Liver smaller\dicom_files\labels'
out_path_labels = 'D:\projects\Tumor Detection\Task03 Liver smaller\\nifi_files\labels'

list_images = glob(in_path_images)
list_labels = glob(in_path_labels)

for patient in list_images:
    patient_name = os.path.basename(os.path.normpath(patient))
    dicom2nifti.dicom_series_to_nifti(patient, os.path.join(out_path_images, patient_name+'.nii.g'))

for patient in list_labels:
    patient_name = os.path.basename(os.path.normpath(patient))
    dicom2nifti.dicom_series_to_nifti(patient, os.path.join(out_path_labels, patient_name+'.nii.g'))

#FIND EMPTY
input_nifti_file_path = 'D:\projects\Tumor Detection\Task03 Liver smaller\\nifi_files\labels\liver_0_0.nii.gz'
list_labels = glob(input_nifti_file_path)

for patient in list_labels:
    nifti_file = nib.load(input_nifti_file_path)
    fdata = nifti_file.get_fdata()
    unique_data = np.unique(fdata)

    if len(unique_data) > 2 == 1:
        print(input_nifti_file_path)
'''