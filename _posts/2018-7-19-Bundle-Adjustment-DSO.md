---
layout: post
title: BA in DSO
---

As we all know DSO is a bundle adjustment in a sliding window, since BA in a whole frame history will enlarge the H matrix for
Gauss-Newton to find the optimal pose estimation. Here I'm going to explore the BA basics and it's relationship to DSO.

Here I define bundle adjustment as a joint optimization process that adjust both the camera pose and the land-marker position.
If you connect the land-marker and the camera, this line is called bundle, they are adjusting the camera pose to make all those
lines (or "bundles") project into another frame with better match, a.k.a, less reprojection error.

Let's model this step by step:

$$P' = Rp + t = [X, Y, Z']^T$$

Here $$(R, t)$$ is the camera pose. And since $$R$$ is an antisymmetric matrix, it has the following shape:

$$\begin{bmatrix} 
a_1 & a_2 & a_3 \\ 
b_1 & b_2 & b_3 \\ 
c_1 & c_2 & c_3  
\end{bmatrix}$$
