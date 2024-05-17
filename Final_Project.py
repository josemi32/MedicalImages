import datetime
import math

import cv2
import imageio
import pydicom
import quaternion
import numpy as np
import matplotlib
import skimage.measure
from matplotlib import pyplot as plt, animation
import scipy
import sys
import os
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import glob
import shutil
from matplotlib.animation import FuncAnimation
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


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




# Orientacion: imageorientation patien (6 valores, las primeras dos columnas )

def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
    """ Apply a colormap to a 2D image. """
    cmap_function = plt.get_cmap(cmap_name)
    img_mapped = cmap_function(img)
    return img_mapped

def visualize_alpha_fusion(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Visualize both image and mask in the same plot. """

    img_sagittal_cmapped = apply_cmap(img,cmap_name='bone')
    # The tab assing different color to each id of the mask
    mask_bone_cmapped = apply_cmap(mask, cmap_name='tab10')
    mask_bone_cmapped = mask_bone_cmapped * mask[..., np.newaxis].astype('bool')


    alpha = 0.2
    # Equation of the alpha fusion
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

    #Normalize max min the img
    return (img - np.min(img)) /(np.max(img) - np.min(img))

def joinImage(img: np.ndarray) -> np.ndarray:

    combined = np.zeros((80,img.shape[1],img.shape[2]))
    for i in range(0,4):
        # Find the indexes where there are something that is not background in the image
        indices = np.where(img[i * 80:(i + 1) * 80, :,:] != 0)
        # Create the id in the matrix
        combined[indices] = i + 1



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
    slices = sorted(slices, key=lambda s: s.SliceLocation, reverse= True)
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


    slice_axial = img3d.shape[2] // 2  # Choose in which coordinate we want to show an image in each palne
    slice_sagittal = img3d.shape[1] // 2
    slice_coronal = img3d.shape[0] // 2

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
    plt.clf()
    #Return the image
    return img3d, pixel_len_mm



def createAnimation(img:np.ndarray,pixel_len_mm:list,n:int,interval:int,nameImg:str,nameGif:str):

    img_min = np.amin(img)
    img_max = np.amax(img)
    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs('results/MIP/', exist_ok=True)
    #   Create projections
    projections = []

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        # Rotate the image alpha degrees on the axial plane
        rotated_img = rotate_on_axial_plane(img, alpha)
        # Proyect the rotation on the Sagittal plane
        projection_sagittal = MIP_sagittal_plane(rotated_img)

        plt.clf()
        plt.imshow(projection_sagittal,cmap='bone', vmin=img_min, vmax=img_max,aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig('results/MIP/' + nameImg +f'_{idx}.png')  # Save animation
        projections.append(projection_sagittal)  # Save for later animation
        plt.clf()
    # Save and visualize animation
    animation_data = [
        [plt.imshow(imggif, animated=True,cmap='bone', vmin=img_min, vmax=img_max,aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for imggif in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                     interval=interval, blit=True)
    anim.save(nameGif + '.gif')  # Save animation
    plt.clf()



def createAnimationAlpha(img: np.ndarray,mask:np.ndarray, pixel_len_mm: list, n: int, interval: int, nameImg: str, nameGif: str):
    img_min = np.amin(img)
    img_max = np.amax(img)
    fig, ax = plt.subplots()
    #   Configure directory to save results
    os.makedirs('results/MIP/', exist_ok=True)
    #   Create projections
    projections = []

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
        #Rotate the image alpha degrees on the axial plane
        rotated_img = rotate_on_axial_plane(img, alpha)
        #Proyect the rotation on the Sagittal plane
        projection_sagittal = MIP_sagittal_plane(rotated_img)

        # Rotate the mask the same degrees as the ct
        rotated_mask = rotate_on_axial_plane(mask, alpha)
        projection_sagittal_mask = MIP_sagittal_plane(rotated_mask)
        # Join the four images in only one image
        #projection_sagittal_mask = joinImage(projection_sagittal_mask)
        # Do the alpha fusion between the two images
        imgrotated = visualize_alpha_fusion(projection_sagittal, projection_sagittal_mask)
        plt.clf()
        plt.imshow(imgrotated, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig('results/MIP/' + nameImg + f'_{idx}.png')  # Save animation
        projections.append(imgrotated)  # Save for later animation
        plt.clf()
        # Save and visualize animation
    animation_data = [
        [plt.imshow(imggif, animated=True, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for imggif in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                                         interval=interval, blit=True)
    anim.save(nameGif + '.gif')  # Save animation
    plt.clf()

def transformImgPacient(img: np.ndarray,shapeRef: tuple,pixR,pixP)->np.ndarray:
    shape_pacient = img.shape
    # pixel_spacing

    pixelSpacing1 = pixP[0] * pixR[0]
    pixelSpacing2 = pixP[1] * pixR[1]


    # Resize the image
    ImgPac = cv2.resize(img,(int(shape_pacient[0] * pixelSpacing1),int(shape_pacient[1]* pixelSpacing2)))
    ImgPac = np.transpose(ImgPac, (2, 0, 1))

    # Crop the Pacient to be the same shape as the reference
    start_x = (ImgPac.shape[2] - shapeRef[2]) // 2
    start_y = (ImgPac.shape[1] - shapeRef[1]) // 2
    start_z = (ImgPac.shape[0] - shapeRef[0]) // 2

    end_x = start_x + shapeRef[2]
    end_y = start_y + shapeRef[1]
    end_z = start_z + shapeRef[0]

    # Crope ROI
    ImgPac = ImgPac[start_z:end_z, start_y:end_y, start_x:end_x]
    return ImgPac



# Similarity function or loss function

def mean_squared_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MSE between two images. """
    return np.mean((img_input - img_reference) ** 2)


# Rigid Motion and all quaternions functions

def translation(
        points: np.ndarray,
        translation_vector: np.ndarray
        ) -> np.ndarray:
    """ Perform translation of points """

    # Simply add the traslation vector to each vector of points in the array
    result = points + translation_vector
    return result


def axial_rotation(
        points: np.ndarray,
        angle_in_rads: float,
        axis_of_rotation: np.ndarray) ->np.ndarray:
    """ Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`. """


    # Calculate the axis rotion and the cos and sin for the quaternion
    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    axis_of_rotation = axis_of_rotation * sin


    # Get the quaternion of rotation
    Quaternion_q = quaternion.as_quat_array(np.insert(axis_of_rotation, 0, cos))





    # Calculate the rotation
    points_rot_quaternion = Quaternion_q * points * quaternion.quaternion.conjugate(Quaternion_q)
    # Get the real part of the quaternions
    points_rot = np.array(quaternion.as_vector_part(points_rot_quaternion))

    # Round the values to the near value
    points_rot = np.round(points_rot).astype(int)


    return points_rot
def translation_then_axialrotation(img:np.ndarray, parameters: tuple[float, ...])-> np.ndarray:
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """

    """

                 Scheme:
                 Create the transformation image with 0
                 Get the indexes,
                 Translate them,
                 Transform them to quaternions,
                 Rotated them,
                 Round them,
                 Filter them,
                 Index the image,
                 Return it

    """

    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters

    # Create the transformed image
    img_trans = np.zeros(img.shape)

    # Get all the points
    points_img = np.argwhere(np.ones(img.shape))
    # Translate the points
    points_traslation = translation(points_img, np.array([t1,t2,t3]))
    # Transform the points to quaternions
    points_as_quaternions = PointsToquartenions(points_traslation)
    #Normalize the axis of rotation
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    axis_rotation = np.array([v1,v2,v3])
    # Rotate the points
    points_rot = axial_rotation(points_as_quaternions,angle_in_rads,axis_rotation)

    # Filter points
    del_indexes = my_filtering_function(points_rot,img.shape)

    points_rot = np.delete(points_rot, del_indexes, axis=0)
    points_img = np.delete(points_img, del_indexes, axis=0)


    # Index the image points
    if np.any(points_rot):
        #for i in range(len(points_img)):
        img_trans[points_img[:, 0], points_img[:, 1],points_img[:,2]] = img[points_rot[:,0],points_rot[:,1],points_rot[:,2]]


    return img_trans




# The inverse of the traslation_then_axialRotation
def axialrotation_then_traslation(img:np.ndarray, parameters: tuple[float, ...])-> np.ndarray:
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """



    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters

    img_trans = np.zeros(img.shape)

    # All the combinations of the indexes
    points_img = np.argwhere(np.ones(img.shape))
    angle_in_rads = -angle_in_rads
    points_as_quaternions = PointsToquartenions(points_img)
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    axis_rotation = np.array([v1, v2, v3])
    # Rotate the points
    points_rot = axial_rotation(points_as_quaternions, angle_in_rads, axis_rotation)

    # Translate the points
    points_traslation = translation(points_rot, np.array([-t1,-t2,-t3]))
    # Filter points

    points_traslation = np.round(points_traslation).astype(int)



    del_indexes = my_filtering_function(points_traslation,img.shape)

    points_traslation = np.delete(points_traslation, del_indexes, axis=0)
    points_img = np.delete(points_img, del_indexes, axis=0)

    if np.any(points_traslation):
        #for i in range(len(points_img)):
        img_trans[points_img[:, 0], points_img[:, 1],points_img[:,2]] = img[points_traslation[:,0],points_traslation[:,1],points_traslation[:,2]]


    return img_trans



# Coregistration using similarity and necesary fucntions



def my_filtering_function(points: np.ndarray,unwanted_value: tuple[int,...])->np.ndarray:
    # Conditions
    condition_1 = (points[:, 0] >= unwanted_value[0]) | (points[:, 0] < 0)
    condition_2 = (points[:, 1] >= unwanted_value[1]) | (points[:, 1] < 0)
    condition_3 = (points[:, 2] >= unwanted_value[2]) | (points[:, 2] < 0)

    #Combine the conditions
    conditions = condition_1 | condition_2 | condition_3

    # indexes
    del_indexes = np.where(conditions)[0]
    return del_indexes




def coregister(ref_img: np.ndarray, inp_img: np.ndarray):
    """ Coregister two sets of images using a rigid transformation. """
    initial_parameters = [
        0, 0, 0,    # Translation vector
        0,          # Angle in rads
        1, 0, 0,    # Axis of rotation
    ]





    def function_to_minimize(parameters):
        """ Transform input image, then compare with reference image."""



        t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters

        print("Inicio transformation")

        print(f'  >> Translation: ({t1:0.02f}, {t2:0.02f}, {t3:0.02f}).')
        print(f'  >> Rotation: {angle_in_rads:0.02f} rads around axis ({v1:0.02f}, {v2:0.02f}, {v3:0.02f}).')



        inp_transf = translation_then_axialrotation(inp_img,parameters)
        print("Fin transformation")


        # Get the error
        mse = mean_squared_error(ref_img, inp_transf)

        print(mse)

        return mse




    result = minimize(function_to_minimize,
                       initial_parameters,
                       method='Powell',
                       options={'disp':True, 'maxiter': 5,'ftol':1e-4}) #xtol y ftol
    return result


def PointsToquartenions (points:np.ndarray)-> np.ndarray:
    # Transform the array of points into an array of quaternions
    quaternion_numpy = quaternion.as_quat_array(np.insert(points,0,0,axis=1))


    return quaternion_numpy



def showImage(img:np.ndarray)-> None:
    #Show the image in each plane
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img[:, :, img.shape[2] // 2], cmap=plt.cm.bone)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img[:, img.shape[1] // 2, :], cmap=plt.cm.bone)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img[img.shape[0] // 2, :, :].T, cmap=plt.cm.bone)
    plt.show()
    plt.clf()

def showImageAlpha(img:np.ndarray,imgref: np.ndarray,cmap:str)-> None:
    # Show the alpha fusion in each plane
    def visualize_alpha_fusion2(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ Visualize both image and mask in the same plot. """

        img_sagittal_cmapped = apply_cmap(img, cmap_name='bone')
        # The tab assing different color to each id of the mask
        mask_bone_cmapped = apply_cmap(mask, cmap_name=cmap)
        mask_bone_cmapped = mask_bone_cmapped * mask[..., np.newaxis].astype('bool')

        alpha = 0.3
        # Equation of the alpha fusion
        img = img_sagittal_cmapped * (1 - alpha) + (mask_bone_cmapped) * alpha

        return img

    axial =visualize_alpha_fusion2(imgref[:, :, imgref.shape[2] // 2],img[:, :, img.shape[2] // 2])
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(axial)

    saggital = visualize_alpha_fusion2(imgref[:, imgref.shape[1] // 2, :], img[:, img.shape[1] // 2, :])
    a2 = plt.subplot(2, 2, 2)

    plt.imshow(saggital)

    coronnal = visualize_alpha_fusion(imgref[imgref.shape[0] // 2, :, :], img[img.shape[0] // 2, :, :])
    a3 = plt.subplot(2, 2, 3)
    plt.imshow(coronnal)

    plt.show()
    plt.clf()




def get_thalamus_mask(img_atlas: np.ndarray) -> np.ndarray:

    #Search for the areas where there are thalamus
    return np.where((img_atlas == 81) | (img_atlas == 82) | np.logical_and(121 <= img_atlas, img_atlas <= 150), 1, 0)




def find_centroid(mask: np.ndarray) -> np.ndarray:

    #Return the centroid of the mask
    return np.mean(np.where(mask == 1),axis=-1).astype(np.int32)


def visualize_thalamus_slices(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        ):
    """ Visualize the slices of the thalamus and the image. """


    a1 = plt.subplot(2, 2, 1)
    axial = visualize_alpha_fusion(img[mask_centroid[0],:,:],mask[mask_centroid[0],:,:])

    plt.imshow(axial)

    a2 = plt.subplot(2, 2, 2)
    saggital = visualize_alpha_fusion(img[:, mask_centroid[1], :], mask[:, mask_centroid[1], :])

    plt.imshow(saggital)

    a3 = plt.subplot(2, 2, 3)
    coronal = visualize_alpha_fusion(img[:, :, mask_centroid[2]], mask[:, :, mask_centroid[2]])

    plt.imshow(coronal)


    plt.show()
    plt.clf()

def visualize_thalamus(img:np.ndarray,thalamus:np.ndarray,nombreImg:str,nombreGif:str)-> None:
    def crop(img: np.ndarray, shapeMask: tuple) -> np.ndarray:


        start_x = (img.shape[2] - shapeMask[2]) // 2
        start_y = (img.shape[1] - shapeMask[1]) // 2
        start_z = (img.shape[0] - shapeMask[0]) // 2

        end_x = start_x + shapeMask[2]
        end_y = start_y + shapeMask[1]
        end_z = start_z + shapeMask[0]

        # Crope ROI
        ImgPac = img[start_z:end_z, start_y:end_y, start_x:end_x]
        return ImgPac
    def createAnimationAlpha2(img: np.ndarray, mask: np.ndarray, n: int, interval: int, nameImg: str,
                             nameGif: str):
        img_min = np.amin(img)
        img_max = np.amax(img)
        fig, ax = plt.subplots()
        #   Configure directory to save results
        os.makedirs('results/MIP/', exist_ok=True)
        #   Create projections
        projections = []

        for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):
            # Rotate the image alpha degrees on the axial plane
            rotated_img = rotate_on_axial_plane(img, alpha)
            # Proyect the rotation on the Sagittal plane
            projection_sagittal = MIP_sagittal_plane(rotated_img)

            # Rotate the mask the same degrees as the ct
            rotated_mask = rotate_on_axial_plane(mask, alpha)
            projection_sagittal_mask = MIP_sagittal_plane(rotated_mask)
            # Join the four images in only one image
            # projection_sagittal_mask = joinImage(projection_sagittal_mask)
            # Do the alpha fusion between the two images
            imgrotated = visualize_alpha_fusion(projection_sagittal, projection_sagittal_mask)
            plt.clf()
            plt.imshow(imgrotated, vmin=img_min, vmax=img_max)
            plt.savefig('results/MIP/' + nameImg + f'_{idx}.png')  # Save animation
            projections.append(imgrotated)  # Save for later animation
            plt.clf()
            print(f"Cargada projecion {idx}")
            # Save and visualize animation
        animation_data = [
            [plt.imshow(imggif, animated=True, vmin=img_min, vmax=img_max)]
            for imggif in projections
        ]
        anim = animation.ArtistAnimation(fig, animation_data,
                                         interval=interval, blit=True)
        anim.save(nameGif + '.gif')  # Save animation



    # Crop the image to be the same size as the thalamus
    ref = crop(img,thalamus.shape)
    centroids = find_centroid(thalamus)
    # Visualize them with the alpha fusion
    visualize_thalamus_slices(ref,thalamus,centroids)
    plt.clf()
    # Do an animation
    createAnimationAlpha2(ref, thalamus, 24, 83, nombreImg, nombreGif)



if __name__ == '__main__':
    # Path to the segmentation and the  CT files





    ### Exercise 1
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

    # Create the animations, for the segmentation, for the ct and for the alpha fusion
    createAnimation(imgseg,pixel_len_mm,24,83,"ProjectionSeg","AnimationSeg")
    createAnimation(imgct, pixel_len_mm, 24, 83, "ProjectionCT", "AnimationCT")
    imgNorm = normalization(imgct)
    imgseg = joinImage(imgseg)
    createAnimationAlpha(imgNorm,imgseg, pixel_len_mm, 24, 83, "ProjectionAlphaFusion", "AnimationAlphaFusion")






    ### Exercise 2
    path_reference = "Dicom Data/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm"
    path_mask ="Dicom Data/AAL3_1mm.dcm"
    path_pacient ="Dicom Data/RM_Brain_3D-SPGR"
    path_oneCtpacient ="Dicom Data/RM_Brain_3D-SPGR/000040.dcm"

    print("Samples per pixel of the Input image:")
    print(load_segmentation(path_oneCtpacient).SamplesPerPixel)
    Ref = load_segmentation(path_reference)
    patOne = load_segmentation(path_oneCtpacient)
    print("Samples per pixel of the reference image:")
    print(Ref.SamplesPerPixel)
    print(Ref.pixel_array.shape) # Shape de 193, 229, 193
    Mask = load_segmentation(path_mask)
    # Get the mask and rotate it
    MaskImg = Mask.pixel_array
    MaskImg = MaskImg[::-1,:,:]
    print("El shape del mask")
    print(Mask.pixel_array.shape)

    # Create the patient
    Pacient,p = createTheCtImage(path_pacient)



    print("Pixel Spacing Phantom: ",Ref.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing)
    print("Pixel spacing patient",patOne.PixelSpacing)


    # Transform the patient to be the same size as the reference
    Pacient = transformImgPacient(Pacient,Ref.pixel_array.shape,Ref.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing,patOne.PixelSpacing)
    print("Shape del paciente")
    print(Pacient.shape)





    # Get the reference image
    ImgRef = Ref.pixel_array


    # Rotate it
    ImgRef = ImgRef[::-1,:,:]


    # Normalize the images to do an alpha fusion later
    Pacient = normalization(Pacient)
    Pacient = gaussian_filter(Pacient, sigma=1)
    ImgRef = normalization(ImgRef)
    # Rotate the patient to be in the same orientation as the reference
    Pacient = Pacient[:,::-1,:]
    #Pacient = np.flip(Pacient,axis = 1)
    showImage(Pacient)
    showImage(ImgRef)
    plt.clf()
    # Visualize the thalamus in the reference space
    thalamus = get_thalamus_mask(MaskImg)
    visualize_thalamus(ImgRef,thalamus,"ThalamusREF","ThalamusREFGif")

    # Coregister the reference and the patient
    result = coregister(ImgRef, Pacient)
    solution_found = result.x

    t1, t2, t3, angle_in_rads, v1, v2, v3 = result.x
    print(f'Best parameters:')
    print(f'  >> Translation: ({t1:0.02f}, {t2:0.02f}, {t3:0.02f}).')
    print(f'  >> Rotation: {angle_in_rads:0.02f} rads around axis ({v1:0.02f}, {v2:0.02f}, {v3:0.02f}).')




    # Transform the input image to the reference space
    img_transformated = translation_then_axialrotation(Pacient,result.x)




    # Show the image transformation
    plt.clf()
    showImage(img_transformated)
    # Show the alpha fusion
    showImageAlpha(img_transformated,ImgRef,"Reds")
    plt.clf()



    # Transform the thalamus to the input space

    Thalamus = axialrotation_then_traslation(thalamus,result.x)

    Thalamus = Thalamus.astype(int)

    # Alpha fusion with the thalamus and the input image

    plt.clf()
    visualize_thalamus(Pacient,Thalamus,"ThalamusInput","ThalamusInputGif")
    

