# README

Alexander Le's 2021 REU Harvard System Biology Internship
Sponsored by:
- Harvard Medical School: Blavatnik Institute
- National Science Foundation
- Simons Foundation
- Harvard QBio

Principal Investigator: Dr. Cengiz Pehlevan
Primary Mentor: Ugne Kilbaite

Link to final presentation slides: [Computational Approaches to Analyze Rat Behavior 
](https://docs.google.com/presentation/d/1nrKaZe0oUiAaghpWq5P3xiBdXXz9smZ-fnYRl-Oz_k8/edit?usp=sharing)

## Introduction
We do not know exactly how the nervous system controls motor behavior in health or disease. Many scientific studies designed to reveal the brain areas, circuits, and chemistry are first conducted in rats, before being conducted in humans. Currently, it is extraordinarily time-intensive to record, interpret, and analyze rat movements 24/7, making it difficult if not impossible to relate these behaviors to neural activity. Therefore I am trying to solve this problem using computational techniques. My current internship project in the Pehlevan Lab is to develop a method for modeling the skeleton using a video feed of animal behavior. There are two major barriers to achieving this goal: (1) we do not always know which 3D data point on the external body correlates to which internal joint; and (2) some data points may not be picked up by the cameras due to environmental noise. Thus, we developing a machine-learning algorithm to predict the name of a 3D data point as well as predict the 3D coordinate of missing points. This algorithm consists of 3 major components: a convolutional neural network (CNN) for initial prediction of points, a graph neural network (GNN) for temporal prediction of points, and a variational autoencoder (VAE) to predict the coordinates of missing points. The code in this notebook consists of the CNN used to predict existing points. Although there is existing software that can help determine the 3D points in space, the code is not optimized for rat behavior and does not account for missing points. Furthermore, we are unable to modify the code to fit different needs since it is proprietary. My research will develop an accessible tool that will enable the analysis of real-time rat behavior without time-intensive user input. We are in a good position to develop this program because we have access to large amounts of rat movement recordings and a robust computing system that can process this large data. As a consequence of our work, many labs will also be able to more powerfully and efficiently determine how mutations, neurological conditions, environmental exposures, or medical interventions affect rat movement and behavior, generating insights and strategies for improving human health.

## Code Introduction
The following code constructs a CNN that can predict the identity of a point in a given frame. The following notebook is divided into two major sections. The first section is to create a CNN to train and test a deep learning model using TensorFlow. The second section is to take temporal data and predict a cluster of points in time. At the end of the program, you can visualize a rat's movement in 3D space. This is a deep learning project analyzing rat behavior and their correlation with neurological activity. Utilized MATLAB to preprocess raw video data and developed Python scripts utilizing TensorFlow to construct a convolutional neural network to identify unmarked 3-dimensional points using spatial and temporal data. Developed a variational autoencoder to predict coordinates of missing joints not registered by raw video feed.

Python Juyper Notebook located in file `Python_Files/skeleton.ipynb `

![ezgif-2-a5fccb3b33](https://user-images.githubusercontent.com/29731342/157175191-f11c1297-fe2e-40c6-81c9-c4b01f0fb16c.gif)


