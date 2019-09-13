---
layout: post
title: monocular depth estimation reviews
---
Several good papers about the monocular depth estimation.
Supervised Depth Estimator:

Using depth ground truth as label to compute pixel wise loss. 


Unsupervised Depth Estimator:
Photometric
(Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction)[https://paperswithcode.com/paper/unsupervised-learning-of-monocular-depth-1] CVPR2018

Our proposal:
Use depth estimator to initialize Visual Odometry, and use visual odometry to refine the depth estimated by the depth estimator.
