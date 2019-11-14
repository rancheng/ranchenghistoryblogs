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

From [DSO](http://vladlen.info/papers/DSO.pdf) (2.2 Model Formulation, page 4)

> evaluating the SSD over such a small neighborhood of pixels is similar to adding first- and second-order irradiance derivative constancy terms (in addition to irradiance constancy) for the central pixel.

Same Paragraph as the above quote. They claim that the 8 pixels pattern is robust to motion blur:

> Our experiments have shown that 8 pixels, arranged in a slightly spread pattern give a good trade-off between computations required for evaluation, robustness to motion blur, and providing sufficient information.

Indeed, their residual pattern works fairly well for the harsh videos, we can observe the results in the following video:

[![DSO Eval](https://img.youtube.com/vi/ymI3FmwU9AY/0.jpg)](https://www.youtube.com/watch?v=ymI3FmwU9AY "DSO Eval")

And we have tested the residual pattern in our module for a small patch (15x15). Note that DSO's pattern strategy choose 8 pattern for their Intel SSE accumulator are only able to hold 8 threads, thus they killed one point on the right bottom corner. In our experiments, we fill that in since we are doing all these in GPU.

![projection_closeup.png]({{site.baseurl}}/images/projection_closeup.png)

The image above shows how the residual pattern works on the ground truth reprojected point, you can see even though the point is perfectly aligned, the local region still exist some noise due to the surface reflection and occlusions, this will introduce extra error into the loss manifold and slowly corrupt the system, in the paper of DSO, the author use the huber weight to normalize the error:

$$E_{pj} = \sum_{p \in N_p} w_p \parallel (I_j[p'] - b_j) - \frac{t_je^{a_j}}{t_ie^{a_i}}(I_i[p] - b_i)\parallel_r$$

Here $$\parallel \cdot \parallel_r$$ is the huber weights, implementation in `DSO` is as following:

```cpp
float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
// huber weight of residual. -> inverse propotional to residual if above threshold.
float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual); // hw is always 0-1
// energy is the huber normalized residual.
energy += hw * residual * residual * (2 - hw);
```
Here the residual is single pixel's intensity error after exposure affine correction, the `r2new_aff[0]` is thus $$ \frac{t_je^{a_j}}{t_ie^{a_i}} $$ and `r2new_aff[1]` is $$b_i - b_j$$. The affine variables $$a_i$$, $$a_j$$, $$b_i$$, $$b_i$$ are jointly optimized in the estimation of the camera pose step. That's why the DSO's hessian block is 8x8, since they have 8 dimensional residuals, 6 DoF pose plus 2 affine models (8x(6+2)).

Let's go back to our first image, where point in the frames with stable illumination to test how DSO's photometric error works. We took the reprojected point, and calculate the photometric error along it's neighbourhood (50x50 and 15x15).

![residual_manifold_reproj.png]({{site.baseurl}}/images/residual_manifold_reproj.png)

Here `reproj_p` is the reprojected point in the target frame with the initial estimated depth and pose, `gt_p` is the ground truth point corresponding to the orignal point in the host frame. From the image above, we can see that the error manifold is never convex even in the small patch (15x15) neighbourhood. Since the pattern size is pretty small, thus this error is qeuivalent to reduced mean convolve of the 3x3 kernel with the whole image. That's why you can still see the rough shape of the original image.

Since DSO is using Gauss-Newton method to do bundle adjustment, thus they also requires the good initialization and locally convex of photometric error, thus they select the points with the higher gradient points as candidates, since the high gradient points neighbourhood will create unique combination of pattern which limited the manifold from decaying their convexity. However, in our method, we are planning to do dense depth estimation, this simple pattern strategy won't hold our locally convex assumption, since there must be repeated textures or low texture patterns.

One of our strategy is to leverage those fixed map point as extra constrains on the optimization of the rest points, by doing this, we are assuming all points are not independently hold their own inverse depths, instead, their inverse depths should be a likelihood that are defined by it's neighbourhood, we can thus propagate our likelihood from the known map points to those unknow points with the update of new observations (constrains). However, this method requires updating the dense map with many iterations until depth convergence, which is super computation expensive.

Thus we propose a more robust and simple method to capture the photometric error efficiently yet still keep the local convexity: random dynamic pattern.

```python
random_pattern = gradius*np.random.rand(gsamples, 2) - gradius/2
```
Here the `gradius`, `gsamples` variables are inverse propotional to the local gradient score.

$$p_g = \frac{\sum_{i \in N}{|I_x(i)| + |I_y(i)|}}{N}$$

Here $$p_g$$ is the local gradient score in point p.

$$g_r = \frac{\eta}{p_g}$$ is the `gradius`, and $$\eta$$ is the normalize constant. Same equation for the `gsamples` with different sample size normalizer.

The effect of the random pattern in the same reprojected point pairs is shown in the following figure:

![error_manifold_same_size.png]({{site.baseurl}}/images/error_manifold_same_size.png)

We only visualized five runs of same random sample radius size, from the figure above, we can see that this manifold is still not smooth enough nor convex. Note that when random sample radius size is small as 3, the error manifold is no difference with the DSO's residual pattern, that is the point distribution will cover most of the pattern in the 8 point pattern.

However, if we change the radius of random pattern, the error manifold will have drastic changes on the local convexity:

![error_manifold_different_rand_size.png]({{site.baseurl}}/images/error_manifold_different_rand_size.png)
