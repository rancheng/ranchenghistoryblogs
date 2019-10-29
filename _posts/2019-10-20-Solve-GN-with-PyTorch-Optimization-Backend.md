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

$$h(I_{t'}, \xi_1, d_2) = K\xi_1K^{-1}d_{2, i}[p_i] \forall p_i \in I_{t'}$$

Let's take any sample point from selected points in my previous post: [gradient based pixel selector](https://rancheng.github.io/gradient-pixel-selector/)

```python
x_ = ptn_x[6488]
y_ = ptn_y[6488]
z_ = visualizer_map[x_, y_]*10
p = torch.autograd.Variable(torch.tensor([z_], dtype=torch.double)).requires_grad_()
```

Here we use `autograd.Variable` to define our point since we are going to optimize it out later on in `LBFGS` loop. Note that you need to apply `requires_grad_()` function in the end since we need this variable in the leaf node of the computation graph, otherwise optimizer won't recognize it.

```python
pxy = torch.tensor([x_, y_]).double()
pxyz = torch.cat((pxy, p)).requires_grad_()
```
```
203 925 11.468850374221802
```
Since the only thing we need to estimate is depth, thus `x_` and `y_` are regarded as constants for each point in the depth map. We extend the depth into 3D point w.r.t to current camera coordinate, and then apply the inverse intrinsic to project into world coordinate:

```python
p2 = torch.reshape(pxyz, (3, 1)).double().requires_grad_()
K = torch.tensor(data.calib.K_cam3, dtype=torch.double)
T = torch.tensor(np.dot(pose_next, np.linalg.inv(pose_current)))
p2 = torch.mm(K.inverse(), p2).requires_grad_()
ext_homo = torch.ones(1,1).double()
p2 = torch.cat((p2, ext_homo), dim=0).requires_grad_()
p2 = torch.mm(torch.tensor(np.linalg.inv(pose_current)), p2).requires_grad_()
p2 = torch.mm(torch.tensor(pose_current), p2).requires_grad_()
```

Here's K and T:

```
K:
tensor([[721.5377,   0.0000, 609.5593],
        [  0.0000, 721.5377, 172.8540],
        [  0.0000,   0.0000,   1.0000]], dtype=torch.float64)
T:
tensor([[ 1.0000e+00, -5.6832e-04, -4.9625e-04, -1.3047e+00],
        [ 5.6926e-04,  1.0000e+00,  1.8940e-03,  4.8494e-01],
        [ 4.9517e-04, -1.8943e-03,  1.0000e+00,  2.3273e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
       dtype=torch.float64)
```

Here p2 is reprojected into the camera coordinate in the target frame:

```
tensor([[-9.4076],
        [-1.4655],
        [11.4689],
        [ 1.0000]], dtype=torch.float64, grad_fn=<MmBackward>)
```

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
