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
0 & -a_1 & a_2 \\ 
a_1 & 0 & -a_3 \\ 
-a_2 & a_3 & 0  
\end{bmatrix}$$

You can see that there's only three variables in the matrix above, thus there's only 3 degree of freedom in rotation, and you
only need to optimize 3 variables in your Gauss-Newton for rotation matrix estimation. This key idea then convert into another
algebra called **lie algebra**, since lie algebra will take another post to introduce, here I'll just explain what it is used
for: it's a group of matrice with customized operations on it. To be understandable, they are just compressing the $$R_{3x3}$$ and $$t_{3x1}$$ into a vector $$\xi_{1x6}$$. The formal definition is as following:

$$

  \begin{align}
    \xi &= \begin{bmatrix}
           \rho \\
           \phi \\
         \end{bmatrix} \in R^6, \rho \in R^3, \phi \in SO(3), \xi^{\wedge} = \begin{bmatrix}
                                                                             \phi^{\wedge} \rho \\
                                                                             0^T 0 \\
                                                                           \end{bmatrix} \in R^{4x4}
  \end{align}   
$$

Here $$SO(3)$$ is the antisymmetric matrix above, the rotation matrix. $$\wedge$$ is the operator that convert a vector into antisymmetric matrix. $$\xi$$ is in $$R^{4x4}$$ means it's under homogeneous coordinate. The reason to extend the 3x3 matrix
into 4x4 homogeneous coordinate is that by introducing a new dimension, we can easily write the **Transformation** linearly.

Okay, enough background introducing. Let's go back to the projection model:

$$
P_c = [u_c,v v_c, 1]^T = [X'/Z', Y'/Z', 1]^T
$$

Equation above is just a normalization, and now we get:

$$
\begin{cases}
  u_c' = u_c(1+k_1r_c^2 + k_2r^4_c)\\
  v_c' = v_c(1 + k_1r_c^2 + k_2r_c^4)
\end{cases}
$$

Here $$k$$ and $$r$$ are the camera skew model factors. This step calculate the rectified coordinate in camera.

Finally, according to the intrinsic model, we can get the final pixel coordinate:

$$
\begin{cases}
 u_s = f_xu_c' + c_x
 v_s = f_xv_c' + c_y
\end{cases}
$$

After projection, we can get the error from the observation points, their pixel coordinates are defined as $$z = [u_o, v_o]^T$$.
Then error can be written as:

$$
e = z - h(\xi, p)
$$

Here h is the projection matrix, p is the 3d point (landmarker) in the world. By sum up all land marker in all time, we get a 
cost function like this:

$$
\sum_t\sum_p{||e_{tp}||^2}
$$

Which is a least squred optimization problem now.
