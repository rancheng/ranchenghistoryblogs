---
layout: post
title: solve BA with PyTorch continued
---

In last post, I've started the trial of solving the Bundle Adjustment problem with PyTorch, since PyTorch's dynamic computation graph and customizable gradient function are very suitable to this large optimization problem, we can easily encode this problem into a learning framework and further push the optimization results into updating the depth estimations and pose estimations in a unsupervised fashion, the true beauty of this method is it introduced more elegant mathemtics to constrain the dimensionality of manifold compared to the direct photometric warping on the whole image. Plus, the sparse structure are extremely fast even apply on cpu with limited hardware (testing on nvidia tx2, cpu only).

Since we have solved pose in last post, now, we're focusing on updating the depth for each active point hessian. And before we start, let's see how this pipeline works:

![reproj_diagram.png]({{site.baseurl}}/images/reproj_diagram.png)

Assume we know the pose already (solved by last post), and we have a very noisy initial of depth from monodepth2, then let's reproject a point from host frame into target frame:

$$p' = K T_i dK^{-1}p $$

Here $$p$$ is the point in host frame, $$p'$$ is the point in target frame, both are in pixel coordinates, and $$K$$ is intrinsic, $$T_i$$ is host to target 3x4 traformation matrix which can be expreseed in SE3, a 1x6 vector. $$d$$ is the depth of point $$p$$, we put it outside $$p$$ since we are going to apply it's depth after it's recovery into world coordinate. 

Since our depth estimation is off, so the projection will be slightly off too, results on kitti will be like this:

![reproj_kitti.png]({{site.baseurl}}/images/reproj_kitti.png)

Let's zoom in:

![reproj_kitti_close.png]({{site.baseurl}}/images/reproj_kitti_close.png)

Now you can see that this reprojection is kind of 6 pixels below the groud truth. I manually piecked a higher gradient area for illustration, however, there are large portion of low gradient regions, for those part of image, we can use the nearby boundary to approximate that area as a local plane and recover depth use this plane.

