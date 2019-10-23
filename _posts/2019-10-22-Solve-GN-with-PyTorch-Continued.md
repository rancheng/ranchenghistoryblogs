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

![reproj_kitti_close_up.png]({{site.baseurl}}/images/reproj_kitti_close_up.png)

Now you can see that this reprojection is kind of 6 pixels below the groud truth. I manually piecked a higher gradient area for illustration, however, there are large portion of low gradient regions, for those part of image, we can use the nearby boundary to approximate that area as a local plane and recover depth use this plane.

we are solving this depth by find the gradient of it w.r.t the reprojection error, namely, photometric error.

$$p' = KT_{h2t}\begin{bmatrix}
\frac{1}{f_x} & 0 & -\frac{c_x}{f_x} \\
0 & \frac{1}{f_y} & -\frac{c_y}{f_y} \\
0 & 0 & 1
\end{bmatrix}\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}$$

let's expand this further more:

$$p' = K\begin{bmatrix}
r_{00} & r_{01} & r_{02} & t_{0} \\
r_{10} & r_{11} & r_{12} & t_{1} \\
r_{20} & r_{21} & r_{22} & t_{2}
\end{bmatrix}\begin{bmatrix}
\frac{x - c_xz}{f_x} \\
\frac{y - c_yz}{f_y} \\
z \\
1
\end{bmatrix}$$

and continue to expand:

$$p' = K\begin{bmatrix}
c_x(t_2 + r_{22}z + r_{20}(\frac{x - c_xz}{f_x}) + r_{21}(\frac{y - (c_yz)}{f_y}) + f_x(t_0 + r_{02}z + r_{00}(\frac{x - c_xz}{f_x}) + r_{01}(\frac{y - (c_yz)}{f_y})) \\
c_y(t_2 + r_{22}z + r_{20}(\frac{x - c_xz}{f_x}) + r_{21}(\frac{y - (c_yz)}{f_y}) + f_y(t_1 + r_{12}z + r_{10}(\frac{x - c_xz}{f_x}) + r_{11}(\frac{y - (c_yz)}{f_y})) \\
t_2 + r_{22}z + r_{20}(\frac{x - c_xz}{f_x}) + r_{21}*(\frac{y - (c_yz)}{f_y})
\end{bmatrix}$$

Now this equation will be very long that's bad for display, let's consider backward:

$$e = \sqrt{(I(p') - I(p))^2}$$

$$\frac{\partial e}{\partial z} = \frac{\partial e}{\partial I} \frac{\partial I}{\partial p}\frac{\partial p}{\partial z} $$

to be more clear the chain should look like this:

$$\Delta_d = \frac{\partial E}{\partial I_{p'}}\frac{\partial I_{p'}}{\partial p'}\frac{\partial p'}{\partial p}\frac{\partial p}{\partial d}$$

Okay, that's simple right? current depth of p is mostly propotational to the product of error and local gradient and the last guess depth of p.

Now, let's focus on the easiest part, solving depth from the selected point hessians (with high local gradient), the `PyTorch` implementation is easy, since autograd function will take care most part of the gradient, the only thing we need to care for is to implement the backward gradient function of $$I(p)$$:



