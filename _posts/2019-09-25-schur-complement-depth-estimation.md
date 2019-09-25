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