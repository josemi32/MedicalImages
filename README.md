DICOM Image Management Project
Introduction
This repository contains the code for a project aimed at  the management of DICOM (Digital Imaging and Communications in Medicine) images. The project focuses on three main aspects: loading, visualization, and coregistration of DICOM images.

Project Overview
The project is divided into two main sections:

DICOM Loading and Visualization: This section focuses on loading patient datasets and visualizing CT images along with segmentation overlays. It includes processes for rearranging DICOM images for correct orientation, and creating animations to showcase segmentation overlays.

3D Rigid Coregistration: Here, the project addresses the alignment of input images with a reference space. Techniques such as image rescaling and rigorous coregistration methodologies are employed to ensure precise alignment, facilitating comprehensive analysis and comparison.

Contents
The repository contains the following main files and directories:

Corresponded CT/ Here there are the images of the CT.
AnimationAlphaFusion is a gif with the alpha fusion between the segmentation and the ct images
AnimationCt is only the animation of the ct images
AnimationSeg is the animation of the segmentation
FinalProject is the python file with the implementations
ReferenciaGif is a gif with the phantom reference
ThalamusInputGif is a gif with the thalamus in the input Space
ThalamusReferenceGif is a gif with the thalamus in the reference Space
