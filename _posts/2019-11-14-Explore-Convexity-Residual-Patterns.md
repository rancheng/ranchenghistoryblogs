---
layout: post
title: Explore the Convexity of Photometric Loss
---

As we can see from my last post [BA with PyTorch](https://rancheng.github.io/Solve-GN-with-PyTorch-Optimization-Backend/) that the direct method that compare the pixel intensity or small patch is extremely non-convex, thus become a huge obstacle for second order optimizers to converge into global or even local minimals. So, in this post we are exploring the method to constrain the photometric error to be as convex as possible. Actually, with a good initialization (achieved by deep methods) the depth estimation can be very close to the ground truth.

This good initialization loose our assumption to be maintaining a locally convex error manifold, since our optimizer will normally terminate in few iterations (estimations are close to the ground truth).

![reprojection_vis.png]({{site.baseurl}}/images/reprojection_vis.png)

Image above illustrate how a single pixel point get reprojected from one frame to another, left image is the host frame and the right image is the target frame. The depth and camera poses estimated from MonoDepth2 are very close to the ground truth which project the point to a location near to the ground truth point, our mission is to adjust the depth to make this reprojection as close to the ground truth point as possible. Since our variable is just depth, thus we can treat this problem as a constraint optimization problem, where the constraint variable is our pose.

![depth_optimize_reproj.png]({{site.baseurl}}/images/depth_optimize_reproj.png)

By adjust the depth, our reprojection will be moving along the tangent line bewteen the epipolar plane and image plane. In the image above, green dot depth is `4m` and blue dot depth is `15m`, and the maximum depth is `40m`. So as you can see, the bigger depth can't change much in the pixel coordinates, and will eventually converge into the point close to the ground truth, this constraint shares a very good property for the optimizer, since it cut the 2D local manifold into a strong convex line that further reduce the computation complexity.

> evaluating the SSD over such a small neighborhood of pixels is similar to adding first- and second-order irradiance derivative constancy terms (in addition to irradiance constancy) for the central pixel.

> Our experiments have shown that 8 pixels, arranged in a slightly spread pattern give a good trade-off between computations required for evaluation, robustness to motion blur, and providing sufficient information.
