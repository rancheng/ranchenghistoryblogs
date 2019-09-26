---
layout: post
title: schur complement for GN Optimization
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

Here $$0_{2 \times 6}$$ means a 2 by 6 zero matix, since $$\frac{\partial e}{\partial \xi}$$ is essentially a $$2 \times 6$$ matrix. 6 means $$R, t$$ which is 6 DoF, what 2 comes from? okay, good question, still remember our observation function in [BA in DSO](https://rancheng.github.io/Bundle-Adjustment-DSO/), the error is $$e = z' - h(\xi, p)$$, here $$e$$ is the error of 2D point position reprojection error! Now you can figure out why the $$\frac{\partial e}{\partial p}$$ is a $$2 \times 3$$ matrix hah? Since $$p$$ is a $$1 \times 3$$ column vector.

From the figure above, we can see that this Jacobian matrix is mostly zero, except two non-zero partial derivatives. Those zero blocks means that the error $$e$$ has no correlation with those poses and land mark points. Thus when apply 
$$H = J^TJ$$, we can see that sparcity from $$J$$ will directly contribute into $$H$$.

If we represent camera poses and land mark points as nodes, the edges that connect each other should be the non-zero entries on Jacobian matrix.

![covisibility.png]({{site.baseurl}}/images/covibility.png)

And you can observe the jacobian with this kind of shape:

![jacobian_block.png]({{site.baseurl}}/images/jacobian_block.png)

The gray area are the non-zero parts, which means there's a partial derivative w.r.t that parameter. (longer gray rectangle is camera pose, gray square is land mark point)

![hessian_arrow.png]({{site.baseurl}}/images/hessian_arrow.png)

Hessian matrix, as you can see, is exactly the same shape as **adjacency matrix** (except the diagonal blocks). We can regard those off-diagonal non-zero blocks as constraints between pose and points, thus we can definitely leverage the sparsity of the $$H$$ matrix and solve pose, point delta updates with schur elimination:

Consider the following Gauss-Newton normal equation:

$$
\begin{bmatrix} 
F^TF & F^TE \\ 
E^TF & E^TE \\  
\end{bmatrix}\Delta x = [F, E]^Te 
$$

We rewrite $$H$$ matrix with 3 blocks $$B$$, $$C$$, $$E$$:

$$
\begin{bmatrix} 
B & E \\ 
E^T & C \\  
\end{bmatrix}\begin{bmatrix} 
\Delta \xi \\ 
\Delta p \\  
\end{bmatrix} = \begin{bmatrix} 
v \\ 
w \\  
\end{bmatrix} 
$$

$$B$$ is the diagonal block matrice, which represent camera pose, $$C$$ is also diagoal block matrice, each block is 3x3. Since diagonal block matrice inverse complexity is way easier than the normal matrix, we can only do inverse on those diagonal blocks. Now, let's do Gauss elimination for the equation above:

$$
\begin{bmatrix} 
I & -EC^{-1} \\ 
0 & I \\  
\end{bmatrix}\begin{bmatrix} 
B & E \\ 
E^T & C \\  
\end{bmatrix}\begin{bmatrix} 
\Delta \xi \\ 
\Delta p \\  
\end{bmatrix} = \begin{bmatrix} 
I & -EC^{-1} \\ 
0 & I \\  
\end{bmatrix}\begin{bmatrix} 
v \\ 
w \\  
\end{bmatrix}
$$

Reorder this, we can get:

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

Since $$C$$ is diagonal blocks, so the inverse of C is very easy to solve.

Many people call these steps above marginalization, there's a reason they call it like this:

From probabilistic perspective, we can regard this elimination as decomposite $$p(\Delta \xi, \Delta p)$$ into solving $$p(\Delta \xi)$$ first and then solve $$p(\Delta p)$$, which is equivalent to solve the conditional probability from marginal probability:

$$P(\Delta \xi, \Delta p) = P(\Delta \xi) \cdot P(\Delta p | \Delta \xi)$$

Many VO are using this method to solve this Hessian matrix, e.g. OKVIS, DSO, they are maintaining a local BA in the sliding window, and force update on each KeyFrame to control accumulated error. That's why they are all sparse visual odometries.

**Question!** What if this $$C$$ matrix is not sparse, how do you solve it? 

Eventually we are aiming toward solving the dense 3D situations, one way we can do now is use GPU to do real-time dense matrix decomposition, or, we can use the Deep methods to initialize $$C$$ and, like First Estimation Jacobian technique, we keep this estimation locally constant, and only update the $$C$$ matrix offline. This way, we are not just shrink the size of land mark observation $$C$$, but eliminated it in the sliding window for all. And the only matrix left to solve is just the diagonal block for camera pose, which is $$o(n^3)$$, $$n$$ is just the KeyFrame number in the sliding window. 
