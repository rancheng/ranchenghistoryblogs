---
layout: post
title: A Mathematical proposal for depth estimaiton optimization in direct method.
---

Direct methods normally hold the photometric consistancy assumption, and the depth estimation from direct methods are jointly estimated with camera poses, which composite into a huge $$H$$ matrix.

Recall that in my previous note on [BA in DSO](https://rancheng.github.io/Bundle-Adjustment-DSO/) which derived the partial derivative of photometric error:

$$
||e(x + \Delta x)||^2 = \sum_t\sum_p||e_{tp} + \frac{\partial e}{\partial \xi}\Delta \xi + \frac{\partial e}{\partial p}\Delta p||^2
$$

And we derived the gauss-newton equation for update $$\Delta x$$

$$H \Delta x = b$$

And take $$J^T = [F, E]$$ into $$H$$:

$$
H\Delta x = J^TJ\Delta x = \begin{bmatrix} 
F^TF & F^TE \\ 
E^TF & E^TE \\  
\end{bmatrix}\Delta x = [F, E]^Te 
$$

To fully understand what H looks like we should first look into what Jacobian matrix looks like:

![hessian_block.png]({{site.baseurl}}/images/hessian_block.png)

Here $$0_{2 \times 6}$$ means a 2 by 6 zero matix, since $$\frac{\partial \e}{\partial \xi}$$ is essentially a $$2 \times 6$$ matrix. 6 means $$R, t$$ which is 6 DoF, what 2 comes from? okay, good question, still remember our observation function in [BA in DSO](https://rancheng.github.io/Bundle-Adjustment-DSO/), the error is $$e = z' - h(\xi, p)$$, here $$e$$ is the error of 2D point position reprojection error! Now you can figure out why the $$\frac{\partial e}{\partial p}$$ is a $$2 \times 3$$ matrix hah? Since $$p$$ is a $$1 \times 3$$ column vector.

From the figure above, we can see that this Jacobian matrix is mostly zero, except two non-zero partial derivatives. Those zero blocks means that the error $$e$$ has no correlation with those poses and land mark points. Thus when apply 
$$H = J^TJ$$, we can see that sparcity from $$J$$ will directly contribute into $$H$$.

If we represent camera poses and land mark points as nodes, the edges that connect each other should be the non-zero entries on Jacobian matrix.

![covisibility.png]({{site.baseurl}}/images/covibility.png)

And you can observe the jacobian with this kind of shape:

![jacobian_block.png]({{site.baseurl}}/images/jacobian_block.png)

The gray area are the non-zero parts, which means there's a partial derivative w.r.t that parameter. (longer gray rectangle is camera pose, gray square is land mark point)

