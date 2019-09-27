---
layout: post
title: Deep Schur Complement
---
Let's recall the last post about schur complement in estimating Hessian matrix:

$$ \begin{bmatrix} B-EC^{-1}E^T & 0 \ E^T & C \
\end{bmatrix}\begin{bmatrix} \Delta \xi \ \Delta p \
\end{bmatrix} = \begin{bmatrix} v-EC^{-1}w \ w \
\end{bmatrix} $$

it's obvious that first line of this equation is irrelevant to $$\Delta p$$. So we can use first line of the equation to solve camera pose:

$$[B - EC^{-1}E^T]\Delta \xi = v - EC^{-1}w$$

and use camera pose we just solved above to further solve $$\Delta p$$:

$$\Delta p = C^{-1}(w - E^T\Delta \xi)$$

Notice this C is the size of all land marks, normally, the number will be kept below 2k to avoid extra computation cost.

What if we want to estimate the dense map from the image and keep this dense map estimation precise and real-time at the same time.

Well, one way comes handy is deep methods, think about it, the landmark variables we are estimating now is just 1d, which is the depth,
what if we give this depth an initial guess in a local frame, and update the upcoming depth from the pose estimated later on,
and this huge dense H matrix will be peeled into a small diagonal matrix that's only contains the camera pose part.
