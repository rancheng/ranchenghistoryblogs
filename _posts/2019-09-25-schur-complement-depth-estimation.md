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

Here $$0_{2 \times 6}$$ means a 2 by 6 zero matix, since $$\frac{\partial \e}{\partial \xi}$$ is essentially a $$2 \times 6$$ matrix. 6 means $$R, t$$ which is 6 DoF, what 2 comes from? okay, good question, still remember our observation function in [BA in DSO](https://rancheng.github.io/Bundle-Adjustment-DSO/), the error is $$e = z' - h(\xi, p)$$, here $$e$$ is the error of 2D point position reprojection error! Now you can figure out why the $$\frac{\partial \e}{\partial p}$$ is a $$2 \times 3$$ matrix hah? Since $$p$$ is a $$1 \times 3$$ column vector.
