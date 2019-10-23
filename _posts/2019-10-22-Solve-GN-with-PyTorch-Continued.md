---
layout: post
title: solve BA with PyTorch continued
---

In last post, I've started the trial of solving the Bundle Adjustment problem with PyTorch, since PyTorch's dynamic computation graph and customizable gradient function are very suitable to this large optimization problem, we can easily encode this problem into a learning framework and further push the optimization results into updating the depth estimations and pose estimations in a unsupervised fashion, the true beauty of this method is it introduced more elegant mathemtics to constrain the dimensionality of manifold compared to the direct photometric warping on the whole image. Plus, the sparse structure are extremely fast even apply on cpu with limited hardware (testing on nvidia tx2, cpu only).

Since we have solved pose in last post, now, we're focusing on updating the depth for each active point hessian. And before we start, let's see how this pipeline works:

Assume we know the pose already, and we have a bad initial of depth from monodepth2.
