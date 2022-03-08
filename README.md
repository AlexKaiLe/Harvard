# README

Alexander Le's 2021 REU Harvard System Biology Internship
Sponsored by:
- Harvard Medical School: Blavatnik Institute
- National Science Foundation
- Simons Foundation
- Harvard QBio

Principal Investigator: Dr. Cengiz Pehlevan
Primary Mentor: Ugne Kilbaite

Link to final presentation slides: [Computational Approaches to Analyze Rat Behavior](https://docs.google.com/presentation/d/1nrKaZe0oUiAaghpWq5P3xiBdXXz9smZ-fnYRl-Oz_k8/edit?usp=sharing)

![ezgif-2-d5c525f9d9](https://user-images.githubusercontent.com/29731342/157276383-91326c2f-c7a6-41d4-a156-6e0255f50cdb.gif)



## Introduction
We do not know exactly how the nervous system controls motor behavior in health or disease. Many scientific studies designed to reveal the brain areas, circuits, and chemistry are first conducted in rats, before being conducted in humans. Currently, it is extraordinarily time-intensive to record, interpret, and analyze rat movements 24/7, making it difficult if not impossible to relate these behaviors to neural activity. Therefore I am trying to solve this problem using computational techniques. My current internship project in the Pehlevan Lab is to develop a method for modeling the skeleton using a video feed of animal behavior. There are two major barriers to achieving this goal: (1) we do not always know which 3D data point on the external body correlates to which internal joint; and (2) some data points may not be picked up by the cameras due to environmental noise. Thus, we developing a machine-learning algorithm to predict the name of a 3D data point as well as predict the 3D coordinate of missing points. This algorithm consists of 3 major components: a convolutional neural network (CNN) for initial prediction of points, a graph neural network (GNN) for temporal prediction of points, and a variational autoencoder (VAE) to predict the coordinates of missing points. The code in this notebook consists of the CNN used to predict existing points. Although there is existing software that can help determine the 3D points in space, the code is not optimized for rat behavior and does not account for missing points. Furthermore, we are unable to modify the code to fit different needs since it is proprietary. My research will develop an accessible tool that will enable the analysis of real-time rat behavior without time-intensive user input. We are in a good position to develop this program because we have access to large amounts of rat movement recordings and a robust computing system that can process this large data. As a consequence of our work, many labs will also be able to more powerfully and efficiently determine how mutations, neurological conditions, environmental exposures, or medical interventions affect rat movement and behavior, generating insights and strategies for improving human health.

## Code Introduction
The following code constructs a CNN that can predict the identity of a point in a given frame. The following notebook is divided into two major sections. The first section is to create a CNN to train and test a deep learning model using TensorFlow. The second section is to take temporal data and predict a cluster of points in time. At the end of the program, you can visualize a rat's movement in 3D space. This is a deep learning project analyzing rat behavior and their correlation with neurological activity. Utilized MATLAB to preprocess raw video data and developed Python scripts utilizing TensorFlow to construct a convolutional neural network to identify unmarked 3-dimensional points using spatial and temporal data. Developed a variational autoencoder to predict coordinates of missing joints not registered by raw video feed.

Python Juyper Notebook located in file `Python_Files/skeleton.ipynb `

## Code Overview
- Don’t know identity of points recorded 
- May include additional points not on rat
- Points could be missing
- Do not know which points are missing and which points are present
- Computing power (time intensive to train/test) 

![Screen Shot 2022-03-08 at 1 19 51 AM](https://user-images.githubusercontent.com/29731342/157276114-41f1a094-7eae-467a-82cf-a1817e384bee.png)

![Screen Shot 2022-03-08 at 1 19 19 AM](https://user-images.githubusercontent.com/29731342/157276127-2bb2ee31-4623-49a7-86d4-8e09072bc89d.png)



Unlabeled Raw 3D skeletal rat points

![ezgif-2-1025e4ee62](https://user-images.githubusercontent.com/29731342/157176132-ecb931d4-413f-44df-a0b3-191dde23024e.gif)

Labeled Raw 3D skeletal rat points

![ezgif-2-c41fd313d5](https://user-images.githubusercontent.com/29731342/157175950-76e4db9f-4ece-417e-898c-67086a651b90.gif)

Mesh connection between every point in the skeleton 

![ezgif-2-383bf9fb8b](https://user-images.githubusercontent.com/29731342/157175616-d2efc4fc-5d23-4cfd-84e4-28ab9f1e61a9.gif)

Highlight correct skeletal structure on top of body mesh and labeled points

![ezgif-2-c2033167fc](https://user-images.githubusercontent.com/29731342/157175759-12d50235-333f-404f-a30f-adbb8e2ef01b.gif)

Outline of correct skeleton with labeled points

![ezgif-2-fafa63b602](https://user-images.githubusercontent.com/29731342/157175524-7244f174-2484-4c62-a2b1-17d387d988e6.gif)

Raw temporal data with missing markers 
![ezgif-2-a5fccb3b33](https://user-images.githubusercontent.com/29731342/157175191-f11c1297-fe2e-40c6-81c9-c4b01f0fb16c.gif)

Prediction on raw temporal data with missing markers of rat skeletal positions 

![ezgif-2-ccdde74a9e](https://user-images.githubusercontent.com/29731342/157175378-fb723542-e01a-4d0d-9303-41427eaca14f.gif)

Variational Autoencoder predictions using predicted labels from temporal and spacial data. Comparing predicted VAE (red) to true position (blue)

![ezgif-2-59e9ce6268](https://user-images.githubusercontent.com/29731342/157177100-bacd7ae8-6221-4548-823b-8998210b9aff.gif)

![loadcat](https://user-images.githubusercontent.com/29731342/157178046-f0883d73-027f-41de-92a7-3c162333255a.gif)

