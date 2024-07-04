# Overview 

This repository contains the assignments for "Computer Vision" (710.120) and "Computergrafik und - vision" (INH.03130UF).

## Assignment Sheet

The assignment sheet is in this git repostory in the doc/ folder.

## Requirements

We build and run the code on an Ubuntu 18.04 machine with the default OpenCV (3.2.0), g++ version (7.4.0) and CMake 3.10.2.
To test your code in this environment you simply need to push your code into the git repository (git push origin master).

Every push will be compiled and tested in our test system. We also store the output and generated images of your program.
As your code will be graded in this testing environment, we strongly recommend that you check that your program compiles 
and runs as you intended (i.e. produces the same output images).
To view the output and generated images of your program, just click on the CI/CD tab -> ``Pipelines''. For every commit,
which you push into the repository, a test run is created. The images of the test runs
are stored as artifacts. Just click on the test run and then you should see the output of your program. On the right 
side of your screen there should be an artifacts link.

We also provide a virtual box image with a pre-installed Ubuntu 18.04: https://cloud.tugraz.at/index.php/s/fGo5qimLH2bqGZz

## Compiling the Code

We use cmake to build our framework. If you are with a linux shell at the root of your repository, just type:

    repopath $ cd src/
    repopath/src $ mkdir build
    repopath/src $ cd build
    repopath/src/build $ cmake ../
    repopath/src/build $ make


To run your program (task1), just type:

    repopath/src/build $ cd ../cv/task1/
    repopath/src/cv/task1/ $ ../../build/cv/task1/cvtask1 0

or:

    repopath/src/build $ cd ../cv/task1/
    repopath/src/cv/task1/ $ ../../build/cv/task1/cvtask1 1

The argument for task1 is just the index of the test-case. As there is just two testcases you might pass 0 or 1 as argument
for the program.

## Making a Submission

To make a submission, you need to create the ``submission'' branch and push it into your repository.
The following commands create a submission branch out the current branch's state and push it to remote. 

    repopath $ git checkout -b submission
    repopath $ git push origin submission

Now you will see the submission branch in your gitlab webinterface. Double check whether it is your final state.
If you are not familiar with git, you'll find a lot of helpful tutorials online. For example this one has a nice 
visualization helping to understand the core concepts: https://learngitbranching.js.org/.

# Side Window Filtering

## Overview

This project involved two main tasks related to point operations and filtering. The tasks included converting images from the BGR color space to the YUV color model and back, visualizing individual Y, U, and V channels, and implementing side window filtering.

## Tasks

### BGR to YUV Conversion

Implemented a function to convert an image from the BGR color space to the YUV color model. This conversion was performed using scaling factors, where the Luma (Y) and Chrominance (U and V) signals were calculated by adding the weighted BGR components. The values were then scaled to fit within the appropriate ranges.

### Visualization of Y, U, and V Channels

Visualized the individual Y, U, and V channels of the YUV image. This involved splitting the image into its respective channels and creating BGR images for each channel. The Y channel was directly converted to BGR, while U and V channels were visualized using colormaps.

### YUV to BGR Conversion

Implemented a function to convert an image from the YUV color model back to the BGR color space. This involved reversing the operations performed during the BGR to YUV conversion, ensuring the values were scaled appropriately.

### Kernel Initialization and Manual 2D Convolution

Initialized filter kernels for various filters including the Gaussian filter. Implemented manual 2D convolution operations using the initialized kernels to apply filters to images.

### Application of Separable Kernels and Convolution

Applied separable kernels to optimize the convolution process. This task involved splitting 2D kernels into two 1D kernels and performing convolution operations more efficiently.

### Implementation of Side Window Box Filter

Implemented the side window box filter to preserve edges while filtering images. This task required splitting the filtering window into multiple smaller windows and selecting the best filter result for each pixel.

### Bonus: Side Window Bilateral Filter

Extended the side window filtering technique to the bilateral filter, which is a non-linear, edge-preserving filter. This involved calculating bilateral filter weights based on both spatial and intensity differences and applying the filter to the image using the side window approach.

