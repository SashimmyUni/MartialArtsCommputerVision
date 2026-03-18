
BoxingHub - v3 2024-08-22 7:40pm
==============================

This dataset was exported via roboflow.com on March 16, 2026 at 10:42 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2347 images.
Jab-cross-hook-uppercut are annotated in YOLO26 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit within)

The following augmentation was applied to create 3 versions of each source image:
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 1.96 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip


