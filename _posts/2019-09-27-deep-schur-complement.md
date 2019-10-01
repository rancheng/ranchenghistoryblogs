---
layout: post
title: Deep Schur Complement
---
Let's recall the last post about schur complement in estimating Hessian matrix:

$$
\begin{bmatrix} 
B-EC^{-1}E^T & 0 \\ 
E^T & C \\  
\end{bmatrix}\begin{bmatrix} 
\Delta \xi \\ 
\Delta p \\  
\end{bmatrix} = \begin{bmatrix} 
v-EC^{-1}w \\ 
w \\  
\end{bmatrix}
$$

it's obvious that first line of this equation is irrelevant to $$\Delta p$$. So we can use first line of the equation to solve camera pose:

$$[B - EC^{-1}E^T]\Delta \xi = v - EC^{-1}w$$

and use camera pose we just solved above to further solve $$\Delta p$$:

$$\Delta p = C^{-1}(w - E^T\Delta \xi)$$

Notice this C is the size of all land marks, normally, the number will be kept below 2k to avoid extra computation cost.

What if we want to estimate the dense map from the image and keep this dense map estimation precise and real-time at the same time.

Well, one way comes handy is deep methods, think about it, the landmark variables we are estimating now is just 1d, which is the depth,
what if we give this depth an initial guess in a local frame, and update the upcoming depth from the pose estimated later on,
and this huge dense H matrix will be peeled into a small diagonal matrix that's only contains the camera pose part.

![hessian_arrow.png]({{site.baseurl}}/images/hessian_arrow.png)

Let's go back to see this Hessian matrix structure, we can find that the bottom right part diagonal block matrix is exactly inverse depth, and the depth is already known for each visible point in two frames (deep depth estimator), now $$C^{-1}$$ is just puting the true depth in each corresponding diagonal cell. This makes the inverse of Hessian matrix in a constant time, and since the initial guess on the correct spot that gradient will be monocular descent (locally convex), this is exactly aligned with the strict assumption on photometric consistancy.

Now consider the photometric consistancy, why can't we use deep method to service as an photometric error term, think about it, the photometric only considers the radiant which is intensity, and normally add another non-linear affine funciton, what if we encode more informations, say, 3 channel colors, and use regression network to learn this color affine model under different light, orientation and distance. Normally the ground truth of depth and poses are already know for Kitti dataset, so, we can build this patch to patch or even pixel to pixel photometric network.

Let's define this photometric network like this:

$$
 E = \theta(p_1, p_2)
$$

![photometric_net.png]({{site.baseurl}}/images/photometric_net.png)

$$\theta$$ takes a small patch of image and output pixelwise error for each pixel. For those pixels without alignment, say, OOB points, the output is infinite and automatically mark them as outlier in the optimization steps.

So our network takes two 3 channels patches, which in total is 6xWxH dimensional data, and output is a WxH matrix. This is simply multiple to 1 process, which compress the data, and network can be designed as the following diagram:

![photometric_net_arch.png]({{site.baseurl}}/images/photometric_net_arch.png)

Our main goal is to minimize the error of predicted reprojection error, confusing hey?

Let's explain by equations:

$$
L_{pe} = || E - E_{true} ||_{huber}
$$

Here $$E$$ is the pixelwise reprojection error prediction, and $$E_{true}$$ is the pixelwised distance to the ground truth reprojection position which can be defined as the following term:

$$
E_{true, ij} = ||I'_{ij} - K(RI_{ij}+t)||_2 
$$

But this will lead us a problem: the unbalanced dataset, due to the mathematic property of reprojection error represented in 2D, we can only get very limited (1) positive examples, this extremely unbalanced dataset will make the proposed network rapidly overfitted by higher error potions.

![convex_reshape.png]({{site.baseurl}}/images/convex_reshape.png)

Eventually, the photo-consistency network reshapes the reprojection errors into a convex manifold. Thus make each candidate pixel can navigate into a global minimal safely.
