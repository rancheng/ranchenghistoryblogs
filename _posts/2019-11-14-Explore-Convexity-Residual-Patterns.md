---
layout: post
title: Explore the Convexity of Photometric Loss
---

As we can see from my last post [BA with PyTorch](https://rancheng.github.io/Solve-GN-with-PyTorch-Optimization-Backend/) that the direct method that compare the pixel intensity or small patch is extremely non-convex, thus become a huge obstacle for second order optimizers to converge into global or even local minimals. So, in this post we are exploring the method to constrain the photometric error to be as convex as possible. Actually, with a good initialization (achieved by deep methods) the depth estimation can be very close to the ground truth.

This good initialization loose our assumption to be simply maintaining a locally convex error manifold, since our optimizer will normally terminate in few iterations (estimations are close to the ground truth).

![reprojection_vis.png]({{site.baseurl}}/images/reprojection_vis.png)

Image above illustrate how a single pixel point get reprojected from one frame to another, left image is the host frame and the right image is the target frame. The depth and camera poses estimated from MonoDepth2 are very close to the ground truth which project the point to a location near to the ground truth point, our mission is to adjust the depth to make this reprojection as close to the ground truth point as possible. Since our variable is just depth, thus we can treat this problem as a constraint optimization problem, where the constraint variable is our pose.

![depth_optimize_reproj.png]({{site.baseurl}}/images/depth_optimize_reproj.png)

By adjust the depth, our reprojection will be moving along the tangent line bewteen the epipolar plane and image plane. In the image above, green dot depth is `4m` and blue dot depth is `15m`, and the maximum depth is `40m`. So as you can see, the bigger depth can't change much in the pixel coordinates, and will eventually converge into the point close to the ground truth, this constraint shares a very good property for the optimizer, since it cut the 2D local manifold into a strong convex line that further reduce the computation complexity.

From [DSO](http://vladlen.info/papers/DSO.pdf (2.2 Model Formulation, page 4)

> evaluating the SSD over such a small neighborhood of pixels is similar to adding first- and second-order irradiance derivative constancy terms (in addition to irradiance constancy) for the central pixel.

Same Paragraph as the above quote. They claim that the 8 pixels pattern is robust to motion blur:

> Our experiments have shown that 8 pixels, arranged in a slightly spread pattern give a good trade-off between computations required for evaluation, robustness to motion blur, and providing sufficient information.

Indeed, their residual pattern works fairly well for the harsh videos, we can observe the results in the following video:

[![DSO Eval](https://img.youtube.com/vi/ymI3FmwU9AY/0.jpg)](https://www.youtube.com/watch?v=ymI3FmwU9AY "DSO Eval")

And we have tested the residual pattern in our module for a small patch (15x15). Note that DSO's pattern strategy choose 8 pattern for their Intel SSE accumulator are only able to hold 8 threads, thus they killed one point on the right bottom corner. In our experiments, we fill that in since we are doing all these in GPU.

![projection_closeup.png]({{site.baseurl}}/images/projection_closeup.png)

The image above shows how the residual pattern works on the ground truth reprojected point, you can see even though the point is perfectly aligned, the local region still exist some noise due to the surface reflection and occlusions, this will introduce extra error into the loss manifold and slowly corrupt the system, in the paper of DSO, the author use the huber weight to normalize the error:

$$E_{pj} = \sum_{p \in N_p} w_p ||(I_j[p'] - b_j) - \frac{t_je^{a_j}}{t_ie^{a_i}}(I_i[p] - b_i)||_r$$

Here \(||.||_r\) is the huber weights, implementation in `C++` is as following:

```cpp
float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
// huber weight of residual. -> inverse propotional to residual if above threshold.
float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual); // hw is always 0-1
// energy is the huber normalized residual.
energy += hw * residual * residual * (2 - hw);
```
Here the residual is single pixel's intensity error after exposure affine correction, the `r2new_aff[0]` is thus \(\frac{t_je^{a_j}}{t_ie^{a_i}}\) and `r2new_aff[1]` is \(b_i - b_j\). The affine variables \(a_i\), \(a_j\), \(b_i\), \(b_i\) are jointly optimized in the estimation of the camera pose step. That's why the DSO's hessian block is 8x8, since they have 8 dimensional residuals, 6 DoF pose plus 2 affine models (8x(6+2)).

