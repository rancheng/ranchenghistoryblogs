---
layout: post
title: Explore Convexity of Residual Patterns
---

As we can see from my last post [BA with PyTorch](https://rancheng.github.io/Solve-GN-with-PyTorch-Optimization-Backend/) that the direct method that compare the pixel intensity or small patch is extremely non-convex, thus become a huge obstacle for second order optimizers to converge into global or even local minimals. So, in this post we are exploring the method to constrain the photometric error to be as convex as possible. Actually, with a good initialization (achieved by deep methods) the depth estimation can be very close to the ground truth.

This prior loose our assumption to be maintaining a locally convex error manifold, since our optimizer will normally terminate in few iterations (estimations are close to the ground truth).

![ba_pipeline_nets.png]({{site.baseurl}}/images/ba_pipeline_nets.png)

Structure from Motion and Visual SLAM applications are heavily dependent on inter-frame geometries, recent deep methods like SfMLearner, MonoDepth, DDVO and many other methods managed to isolate the joint optimization of camera pose and the estimation of depth. They treat the training of pose network as supervised learning, since most datasets offers ground truth camera pose. This trick circumvent the dense optimization problem in BA and extremely simplified the learning life cycle with back warping and direct photometric error losses. However, one can still argue that the parameter manifold of SE3 learnt from pose network can be limited and dataset dependent, see detailed explanation from here: [E2E PoseNet](https://hal.archives-ouvertes.fr/hal-01879117/document). Besides, the pose estimations are always in limited distributions which will be very unfriendly for UAV with large maneuver degrees. To make this pipeline more robust, we introduced our proposal on extend the depth and pose estimation with a windowed Bundle Adjustment Backend.
