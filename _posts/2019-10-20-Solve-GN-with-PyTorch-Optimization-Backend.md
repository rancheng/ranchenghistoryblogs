---
layout: post
title: Solve BA with PyTorch Optimization Backend
---

This post shows how to use LBFGS optimizer to parallelly optimize dense BA.

![ba_pipeline_nets.png]({{site.baseurl}}/images/ba_pipeline_nets.png)

Structure from Motion and Visual SLAM applications are heavily dependent on inter-frame geometries, recent deep methods like SfMLearner, MonoDepth, DDVO and many other methods managed to isolate the joint optimization of camera pose and the estimation of depth. They treat the training of pose network as supervised learning, since most datasets offers ground truth camera pose. This trick circumvent the dense optimization problem in BA and extremely simplified the learning life cycle with back warping and direct photometric error losses. However, one can still argue that the parameter manifold of SE3 learnt from pose network can be limited and dataset dependent, see detailed explanation from here: [E2E PoseNet](https://hal.archives-ouvertes.fr/hal-01879117/document). Besides, the pose estimations are always in limited distributions which will be very unfriendly for UAV with large maneuver degrees. To make this pipeline more robust, we introduced our proposal on extend the depth and pose estimation with a windowed Bundle Adjustment Backend.

Here's a simple illustration on how depth and pose network works:

![ba_pipeline.png]({{site.baseurl}}/images/ba_pipeline.png)

Unlike the traditional depth and pose estimation system which treat pose and depth estimation separately (SfMLearner, MonoDepth2, BANet) we are going to jointly optimize pose $$\xi$$ and dense depth map $$d_i$$ for each BA iteration, from the diagram above you can see we just applied two iterations of dense BA since we are going to make sure this whole pipeline goes real time, even with the help of GPU.

Inspired by [DSO](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7898369), our pipeline is also using coarse to fine fashion. The reason is: pose of consequtive frames captured at normal frequency are normally small enough, thus can be estimated by coarse and small scale of image, with this coarse pose, we are able to refine our initial depth map guess quickly, and then propagate this refined depth map into larger scale space. By this way, we can quickly initialize the depth and pose with a reasonable prior, and use this initial state to optimize will reach a good critical point which has higher change to be global minimum (depth recovering is an ill-pose problem since image manifold is super non-convex, that's why we need a good initialization). Now after convergence of the dense depth map, we will inturn adjust pose again (that's why they call Bundle Adjustment, they are adjusting two end of the bundle projection ray: camera end $$\xi$$, and world end $$d$$). Eventually, we will use them to do backwarping and calculate the photometric loss, smoothness loss, occlusion loss and affine loss.

Now with knowing enough big picture on what we should do, let's take a close look on how to implement them. First step is to estimate pose, which was introduced in my [last post](https://rancheng.github.io/Solve-GN-with-PyTorch/). Then we can do depth estimation with the following equation:

$$h(I_{t'}, \xi_1, d_2) = I_{t'}[KT_{w2c}\xi_1d_{2, i}T_{w2c}^{-1}[p_i]K^{-1}p_i] \forall i \in \theta$$

Here $$\xi$$ is the camera pose and the $$\theta$$ is the selected gradient point sets.

Let's take any sample point from selected points in my previous post: [gradient based pixel selector](https://rancheng.github.io/gradient-pixel-selector/)

```python
v_ = chosen_p[0]
u_ = chosen_p[1]
z_ = visualizer_map[v_, u_]*10
d = torch.autograd.Variable(torch.tensor([z_], dtype=torch.double)).requires_grad_()
```

Here we use `autograd.Variable` to define our point since we are going to optimize it out later on in `LBFGS` loop. Note that you need to apply `requires_grad_()` function in the end since we need this variable in the leaf node of the computation graph, otherwise optimizer won't recognize it.

Since we only care about the depth, so we isolated the point and the depth variable:

```python
pxyz = torch.tensor([u_, v_, 1]).double()
```

`pxyz` tensor's z value is set as 1. We are going to recover the depth later on.

```
tensor([[725.],
        [135.],
        [  1.]], dtype=torch.float64)
```

Since the only thing we need to estimate is depth, thus `u_` and `v_` are regarded as constants for each point in the depth map. We extend the depth into 3D point w.r.t to current camera coordinate, and then apply the inverse intrinsic to project into world coordinate.

```python
pki = torch.mm(K.inverse(), p)
pki = d*pki
```
```python
tensor([[ 4.7520],
        [-1.5582],
        [29.7016]], dtype=torch.float64, grad_fn=<MulBackward0>)
```

Here's $$K$$ and $$T_{w2c}$$:

```
K:
tensor([[721.5377,   0.0000, 609.5593],
        [  0.0000, 721.5377, 172.8540],
        [  0.0000,   0.0000,   1.0000]], dtype=torch.float64)

T_w2c = torch.tensor(t_imu2cam, dtype=torch.double)

T_w2c:
tensor([[ 9.9875e-04, -9.9999e-01,  4.2594e-03, -7.8463e-01],
        [ 8.4169e-03, -4.2508e-03, -9.9996e-01,  7.1945e-01],
        [ 9.9996e-01,  1.0346e-03,  8.4126e-03, -1.0891e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
       dtype=torch.float64)

T_w2c.inverse():
tensor([[ 9.9875e-04,  8.4169e-03,  9.9996e-01,  1.0838e+00],
        [-9.9999e-01, -4.2508e-03,  1.0346e-03, -7.8044e-01],
        [ 4.2594e-03, -9.9996e-01,  8.4126e-03,  7.3192e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
       dtype=torch.float64)
```

With all those known parameters defined, we can now apply the reprojection:

```python
# extend the point to homogeneous coordinate
ext_homo = torch.ones(1,1).double()
pki = torch.cat((pki, ext_homo), dim=0)
-------------------------------------------------------------
output:
tensor([[ 4.7520],
        [-1.5582],
        [29.7016],
        [ 1.0000]], dtype=torch.float64, grad_fn=<CatBackward>)
-------------------------------------------------------------
pTwcki = torch.mm(T_w2c.inverse(), pki)
pTwcki
-------------------------------------------------------------
output:
tensor([[30.7759],
        [-5.4951],
        [ 2.5602],
        [ 1.0000]], dtype=torch.float64, grad_fn=<MmBackward>)
-------------------------------------------------------------
pTTwcki = torch.mm(torch.tensor(pose_current), pTwcki)
pTTwcki = torch.mm(torch.tensor(np.linalg.inv(pose_next)), pTTwcki)
pTTwcki
-------------------------------------------------------------
output:
tensor([[29.3777],
        [-5.4914],
        [ 2.5796],
        [ 1.0000]], dtype=torch.float64, grad_fn=<MmBackward>)
-------------------------------------------------------------
pTTTki = torch.mm(T_w2c, pTTwcki)
pTTTki
-------------------------------------------------------------
output:
tensor([[ 4.7471],
        [-1.5895],
        [28.3036],
        [ 1.0000]], dtype=torch.float64, grad_fn=<MmBackward>)
-------------------------------------------------------------
pKTTTki = torch.mm(K, pTTTki[0:3])
pKTTTki
-------------------------------------------------------------
output:
tensor([[20677.8739],
        [ 3745.5288],
        [   28.3036]], dtype=torch.float64, grad_fn=<MmBackward>)
-------------------------------------------------------------
pKTTTki = pKTTTki/pKTTTki[-1]
-------------------------------------------------------------
output:
tensor([[730.5752],
        [132.3342],
        [  1.0000]], dtype=torch.float64, grad_fn=<DivBackward0>)
```

Here `pKTTTki` is reprojected into the camera coordinate in the target frame:

```
tensor([[730.5752],
        [132.3342],
        [  1.0000]], dtype=torch.float64, grad_fn=<DivBackward0>)
```

Compared to the original coordinate from host frame:

```
tensor([[725.],
        [135.],
        [  1.]], dtype=torch.float64)
```

Let's plot them to check whether this align or not:




Notice that we keep out gradient function hooked in each step, so that the `backward` on loss will eventually navigate us to the exact gradient we want.

Then convert to pixel coordinate with gradients kept:
```python
p2 = torch.mm(K, p2[0:3]).requires_grad_()
p2 = p2/p2[-1]
p2.requires_grad_()
```
we got this:

```
tensor([[17.7001],
        [80.6532],
        [ 1.0000]], dtype=torch.float64, grad_fn=<DivBackward0>)
```

`17` and `80` are our coordinate in the new frame.

Let's have a look on the image to do a sanity check first:
