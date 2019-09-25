---
published: false
---
## A Mathematical proposal for depth estimaiton optimization in direct method.

Direct methods normally hold the photometric consistancy assumption, and the depth estimation from direct methods are jointly estimated with camera poses, which composite into a huge $$H$$ matrix.

Recall that in my previous note on [BA in DSO](https://rancheng.github.io/Bundle-Adjustment-DSO/) which derived the partial derivative of photometric error:

$$ argmin_{\Delta \xi, \Delta p }||e + F\Delta \xi + E\Delta p ||_2 $$ and we derived the gauss-newton equation for update $$\Delta x$$

$$H \Delta x = b$$

And take $$J^T = [F, E]$$ into $$H$$:

$$
H\Delta x = J^TJ\Delta x = \begin{bmatrix} 
F^TF & F^TE \\ 
E^TF & E^TE \\  
\end{bmatrix}\Delta x = [F, E]^Te 
$$

To fully understand what H looks like we should first look into what Jacobian matrix looks like:

![hessian_block.png]({{site.baseurl}}/_posts/hessian_block.png)
