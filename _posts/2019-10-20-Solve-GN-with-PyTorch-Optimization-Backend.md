---
layout: post
title: Solve BA with PyTorch Optimization Backend
---

This post shows how to use LBFGS optimizer to parallelly optimize dense BA.

Structure from Motion and Visual SLAM applications are heavily dependent on inter-frame geometries, recent deep methods like SfMLearner, MonoDepth, DDVO and many other methods managed to isolate the joint optimization of camera pose and the estimation of depth. They treat the training of pose network as supervised learning, since most datasets offers ground truth camera pose. This trick circumvent the dense optimization problem in BA and extremely simplified the learning life cycle with back warping and direct photometric error losses. However, one can still argue that the parameter manifold of SE3 learnt from pose network can be limited and dataset dependent, see detailed explanation from here: [E2E PoseNet](https://hal.archives-ouvertes.fr/hal-01879117/document). Besides, the pose estimations are always in limited distributions which will be very unfriendly for UAV with large maneuver degrees. To make this pipeline more robust, we introduced our proposal on extend the depth and pose estimation with a windowed Bundle Adjustment Backend.

Here's a simple illustration on how depth and pose network works:

![ba_pipeline.png]({{site.baseurl}}/images/ba_pipeline.png)

And this is the pose/depth network:

![ba_pipeline_nets.png]({{site.baseurl}}/images/ba_pipeline_nets.png)

Unlike the traditional depth and pose estimation system which treat pose and depth estimation separately (SfMLearner, MonoDepth2, BANet) we are going to jointly optimize pose $$\xi$$ and dense depth map $$d_i$$ for each BA iteration, from the diagram above you can see we just applied two iterations of dense BA since we are going to make sure this whole pipeline goes real time, even with the help of GPU.

Inspired by [DSO](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7898369), our pipeline is also using coarse to fine fashion. The reason is: pose of consequtive frames captured at normal frequency are normally small enough, thus can be estimated by coarse and small scale of image, with this coarse pose, we are able to refine our initial depth map guess quickly, and then propagate this refined depth map into larger scale space. By this way, we can quickly initialize the depth and pose with a reasonable prior, and use this initial state to optimize will reach a good critical point which has higher change to be global minimum (depth recovering is an ill-pose problem since image manifold is super non-convex, that's why we need a good initialization). Now after convergence of the dense depth map, we will inturn adjust pose again (that's why they call Bundle Adjustment, they are adjusting two end of the bundle projection ray: camera end $$\xi$$, and world end $$d$$). Eventually, we will use them to do backwarping and calculate the photometric loss, smoothness loss, occlusion loss and affine loss.

Our major optimziations are happening in GPU, which allows us dense BA in a local sliding window (5~7 frames).
