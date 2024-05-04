import datetime

import cv2
import imageio
import pydicom
import numpy as np
import matplotlib
import skimage.measure
from matplotlib import pyplot as plt, animation
import scipy
import sys
import os
import glob
import shutil
from matplotlib.animation import FuncAnimation
from PIL import Image
def find_correct_ct(segmentation, ruta_carpeta_ct):
    # Obtain the studyInstanceUid
    uid_serie_segmentacion = segmentation.StudyInstanceUID

    # List for the names of the images
    namesCT = []
    ctarray = []


    for archivo in os.listdir(ruta_carpeta_ct):
        ruta_archivo = os.path.join(ruta_carpeta_ct, archivo)
        if os.path.isfile(ruta_archivo):
            # Load the ct

            ds = pydicom.dcmread(ruta_archivo)



            #break
            # If the studyInstaceUid is the same we append it into the list
            if ds.StudyInstanceUID == uid_serie_segmentacion:
                namesCT.append(archivo)
                ctarray.append(ds)

    return namesCT,ctarray





def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm,angle = angle_in_degrees, axes=(1,2), reshape=False)


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis = 2)

def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis = 1)
def load_segmentation(path):
    # Load the DICOM file
    ds = pydicom.dcmread(path)
    return ds

def show_headers(imgSeg,imgCT):
    print("Headers of the segmentation: ")
    print(imgSeg.dir())

    print("Headers of a specific acquisition: ")
    print(imgCT.dir())






def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
    """ Apply a colormap to a 2D image. """
    cmap_function = plt.get_cmap(cmap_name)
    img_mapped = cmap_function(img)
    return img_mapped

def visualize_alpha_fusion(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Visualize both image and mask in the same plot. """

    img_sagittal_cmapped = apply_cmap(img,cmap_name='bone')

    mask_bone_cmapped = apply_cmap(mask, cmap_name='tab10')
    mask_bone_cmapped = mask_bone_cmapped * mask[..., np.newaxis].astype('bool')


    alpha = 0.2
    img = img_sagittal_cmapped * (1 - alpha) + (mask_bone_cmapped) * alpha

    return img

def is_binary_image(image: np.ndarray) -> bool:

    min_val = np.min(image)
    max_val = np.max(image)
    if min_val == 0 and (max_val == 1 or max_val == 255):
        return True

    # Count unique values
    unique_values = np.unique(image)
    if len(unique_values) == 2:
        return True

    # Types of values
    if image.dtype == bool or image.dtype == np.uint8:
        return True

    return False


def normalization(img:np.ndarray)-> np.ndarray:

    return (img - np.min(img)) /(np.max(img) - np.min(img))

def joinImage(img: np.ndarray) -> np.ndarray:

    combined = np.zeros((80,img.shape[1]))
    for i in range(0,4):
        indices = np.where(img[i * 80:(i + 1) * 80, :] != 0)
        combined[indices] = i + 1
        #combined[img[i*80:(i+1)*80,:] != 0] = i + 1


    combined = combined.astype(np.uint8)


    return combined


def rearrengeSeg(seg)-> np.ndarray:
    # Array for the ordered indexes
    positions = []

    # For each slice in the segmentation we extract the z position of the patient and the organ number
    for element in seg.PerFrameFunctionalGroupsSequence:
        positions.append([element.PlanePositionSequence[0].ImagePositionPatient[2],
                          element.SegmentIdentificationSequence[0].ReferencedSegmentNumber])


    positions = np.array(positions)
    # Order the array by the position and then for the organ, and return the ordered indexes
    Ordered_Indexs = np.lexsort((positions[:, 0], positions[:, 1]))

    # Apply the ordered indexes to the segmentation image
    imgseg = seg.pixel_array[Ordered_Indexs, :, :]
    return imgseg



def loadSlicesCt(path: str) ->list:

    files = []
    for fname in os.listdir(path):
        ruta_archivo = os.path.join(path, fname)
        files.append(load_segmentation(ruta_archivo))

    print("Number of acquisitions: {}".format(len(files)))

    # skip files with no SliceLocation if there is any.
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # Order the slices based on the slicelocation and then reverse it becuase it goes down to the top and we want top to down.
    slices = sorted(slices, key=lambda s: s.SliceLocation, reverse=True)
    return slices

def createTheCtImage(path:str)-> np.ndarray:

    #Load the slices of the ct image
    slices = loadSlicesCt(path)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing # two headers of the ct images,
    ss = slices[0].SliceThickness

    ax_aspect = ps[1] / ps[0] # calculate the aspect of axial in order to show it correctly
    sag_aspect = ps[1] / ss # calculate the aspect of saggital in order to show it correctly based on the thickness
    cor_aspect = ss / ps[0] # calculate the aspect of coronal in order to show it correctly based on the thickness

    pixel_len_mm = [ax_aspect, sag_aspect, cor_aspect] # save this values for the animation

    # create 3D array, first a list with the shape of the images
    img_shape = list(slices[0].pixel_array.shape)

    img_shape.append(len(slices)) # then as a third coordinate the numbers of slices
    img3d = np.zeros(img_shape) # create the 3d image with the shape

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):

        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    # plot 3 orthogonal slices


    slice_axial = 40  # Choose in which coordinate we want to show an image in each palne
    slice_sagittal = 400
    slice_coronal = 400

    # Show the images
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d[:, :, slice_axial], cmap=plt.cm.bone)
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, slice_sagittal, :], cmap=plt.cm.bone)
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d[slice_coronal, :, :].T, cmap=plt.cm.bone)
    a3.set_aspect(cor_aspect)
    plt.show()

    #Return the image
    return img3d, pixel_len_mm



def createAnimation(img:np.ndarray,pixel_len_mm:list,n:int,interval:int,nameImg:str,nameGif:str,combination:bool,mask:np.ndarray = None):

    img_min = np.amin(img)
    img_max = np.amax(img)
    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs('results/MIP/', exist_ok=True)
    #   Create projections
    projections = []

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        rotated_img = rotate_on_axial_plane(img, alpha)
        projection_sagittal = MIP_sagittal_plane(rotated_img)
        imgrotated = projection_sagittal


        if combination:
            rotated_mask = rotate_on_axial_plane(mask,alpha)
            projection_sagittal_mask = MIP_sagittal_plane(rotated_mask)
            projection_sagittal_mask = joinImage(projection_sagittal_mask)
            imgrotated = visualize_alpha_fusion(imgrotated,projection_sagittal_mask)

        plt.imshow(imgrotated, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig('results/MIP/' + nameImg +f'_{idx}.png')  # Save animation
        projections.append(imgrotated)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(imggif, animated=True, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for imggif in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=interval, blit=True)
    anim.save(nameGif + '.gif')  # Save animation





if __name__ == '__main__':
    # Path to the segmentation and the  CT files
    path_segmentation = 'manifest-1713979305387/HCC-TACE-Seg/HCC_008/02-02-1998-NA-CT ABD LIV PRO-01687/300.000000-Segmentation-70014/1-1.dcm'
    path_CT = 'Correspondent CT'

    # Load the segmentation
    segmentation = load_segmentation(path_segmentation)

    # Rearrange the segmentation
    imgseg = rearrengeSeg(segmentation)


    img3d,pixel_len_mm = createTheCtImage(path_CT)

    print("Shape segmentacion")
    print(imgseg.shape)
    print("Shape img3d")
    print(img3d.shape)
    # we transpose the ct image because the shape is (512,512,80) and we want (80,512,512) for the rotation
    imgct = np.transpose(img3d, (2, 0, 1))

    print("Shape transpose")
    print(imgct.shape)  # Print (80, 512, 512)

    createAnimation(imgseg,pixel_len_mm,24,43,"ProjectionSeg","AnimationSeg",False)
    createAnimation(imgct, pixel_len_mm, 24, 43, "ProjectionCT", "AnimationCT", False)
    imgNorm = normalization(imgct)
    createAnimation(imgNorm, pixel_len_mm, 24, 43, "ProjectionAlphaFusion", "AnimationAlphaFusion", False,imgseg)



    """


    projectionsCT = []
    img_minCT = np.amin(imgct)
    img_maxCT = np.amax(imgct)

    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs('results/MIP/', exist_ok=True)
    #   Create projections
    n = 24
    imgctnorm = normalization(imgct)
    #imgctnorm[imgctnorm < 0.3] = 0

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        rotated_imgCT = rotate_on_axial_plane(imgctnorm, alpha)
        rotated_imgSeg = rotate_on_axial_plane(imgseg,alpha)


        projection_saggitalCT = MIP_coronal_plane(rotated_imgCT)
        projection_saggitalSeg = MIP_coronal_plane(rotated_imgSeg)





        combined = visualize_alpha_fusion(projection_saggitalCT,joinImage(projection_saggitalSeg))






        plt.imshow(combined, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig(f'results/MIP/Projection_{idx}.png')  # Save animation
        projectionsCT.append(combined)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True,vmin=img_minCT, vmax=img_maxCT, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for img in projectionsCT
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=43, blit=True)
    anim.save('AnimationCT2.gif')  # Save animation

    plt.show()

    """

