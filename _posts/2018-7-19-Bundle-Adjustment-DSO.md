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
                                                                             \phi^{\wedge} & \rho \\
                                                                             0^T & 0 \\
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

Here h is the projection matrix, p is the 3d point (landmarker) in the world. By sum up all land marker in all time, we get a 
cost function like this:

$$
\sum_t\sum_p{||e_{tp}||^2}
$$

Which is a least squred optimization problem now. Since observation function $$h(\xi, p)$$ is a non-linear function, we can use
Gauss-Newton method to optimize it. The delta update is $$\Delta = [\xi, p]$$ which is a $$6+n$$ vector, 6 is the degree of freedom of camera pose, $$n$$ is the land mark point number. Accordingly, use taylor approximation, we can approximate the cost
function as the following equation:
$$
||e(x + \Deltax)||^2 = \sum_t\sum_p||e_{tp} + \frac{\partial e}{\partial \xi}\Delta xi + \frac{\partial e}{\partial \p}\Delta p||^2
$$

How to solve the partial derivatives $$\frac{\partial e}{\partial \delta\xi}$$ and $$\frac{\partial e}{\partial \p}$$ is now the key for use to find the update gradients, well for projection p, it's easy:

$$
\frac{\partial e}{\partial \p} = \frac{partial e}{\partial P'} \frac{\partial P'}{\partial p} = \begin{bmatrix} 
\frac{f_x}{Z'} & 0 & -\frac{f_xX'}{Z'^2} \\ 
0 & \frac{f_y}{Z'} & -\frac{f_yY'}{Z'^2} \\  
\end{bmatrix}R
$$

Here $$P'$$ is the 3d point in camera coordinate. Since $$P' = Rp + t$$, thus $$\frac{\partial P'}{\partial p} = R$$.

The real headache came from the camera pose $$\xi$$, since it's in SE(3), and it's hard to directly solve partial derivative 
in SE(3), however, since the lie group shares the good property as in eucledian space, they are all smooth manifolds, thus we
can introduce perturbation lemma ([J Huebschmann et, al](https://arxiv.org/pdf/0708.3977)) to approximate the jacobian.

I'll discuss the derivation in Lie group in another post in detail, so if you are not familiar with the lie group, don't be 
afraid, just note that it's just convert from one coordinate system to another coordinate system and mapped the geometric
operation rules.

Here I'm going to give the results, according to perturbation lemma:

$$
\frac{\partial e}{\partial \delta \xi} = \frac{(\delta \xi \oplus \xi)\partial e}{\parial \delta \xi} = \frac{\partial e}{\partial P'} \frac{\partial P'}{\partial \xi}
$$

Here $$\delta \xi$$ is the small perturbation delta and this equation decomposed derivative of lie group into two parts: 
partial derivative to the 3d point and the partial derivative of the perturbation delta, since we know that:

$$\frac{\partial e}{\partial P'} = begin{bmatrix} 
\frac{f_x}{Z'} & 0 & -\frac{f_xX'}{Z'^2} \\ 
0 & \frac{f_y}{Z'} & -\frac{f_yY'}{Z'^2} \\  
\end{bmatrix}$$

Now the second partial is:
$$frac{\partial P'}{\partial \delta \xi} = \frac{\partial (Tp)}{\partial \delta \xi} = begin{bmatrix} 
I & -P'^{\wedge} \\ 
0 & 0 \\  
\end{bmatrix}$$

Note that the matrix above is a 4x4 matrix in homogeneous coordinate, and we only extract the first 3 rows:

$$\frac{\partial P'}{\partial \delta\xi} = [I, -P'^{\wedge}]$$
