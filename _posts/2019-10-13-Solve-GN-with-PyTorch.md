---
layout: post
title: Solve BA with PyTorch
---

Since Bundle Adjustment is heavily depending on optimization backend, due to the large scale of Hessian matrix, solving Gauss-Newton directly is extremely challenging, especially when solving Hessian matrix and it's inverse. Scale of Hessian matrix is depending on number of map points, so to this end, we can first estimate the point inverse depths and then update the poses in two stage manner, however, this requires a very good initializaiton for depth estimation, and requires a convex local photometric loss to make sure positive definite property of the Hessian.

![gn_opt.png]({{site.baseurl}}/images/gn_opt.png)

Figure above illustrated Gauss-Newton method pipeline, which is iteratively estimating $$\Delta x$$ and updating $$x$$ to the convergence point where $$\Delta x$$ close to zero. This pipeline can be well modeled by `PyTorch` where we can easily use `autograd` to solve Jacobians and pseudo-inverse of Hessian matrix.

![ba_nn.png]({{site.baseurl}}/images/ba_nn.png)

```python
# two frames:
hostFrame = frame_window[0]
targetFrame = frame_window[-1]
# define the H matrix
H = torch.randn(M, N)
b = torch.randn(N, 1)
# reshape input H matrix:
x = torch.randn(N, 1, requires_grad=True)
# initial learning rate
lr = 1e-5
optimizer = optim.LBFGS([x], lr=lr)
def closure():
    H, b, loss = calcResAndGS(x)
    optimizer.zero_grad()
    loss.backward()
    return loss
for lr in lr * .5**np.arange(10):
    optimizer.step(closure)
```

Here function `calcResAndGs` is the major function to calculate loss and estimate Hessian and Jacobian matrix from current pose:

```python
def calcResAndGS(x):
    JbBuffer = torch.zeros([x.shape[0], npts], dtype=torch.float32)
    refToNew = sophus.SE3(x)
    # loop all selected points
    e = torch.zeros([npts], dtype=torch.float32)
    for i in range(npts):
        reproj_p = torch.matmul(refToNew, point[i])
        idepth = reprj_p[-1]
        reproj_p = torch.matmul(K, reproj_p)
        pe = targetFrame[reproj_p[0], reproj_p[1]] - hostFrame[point[i][0], point[i][1]]
        e[i] = pe
        dIdx = targetFrame.dx[reproj_p[0], reproj_p[1]]
        dIdy = targetFrame.dy[reproj_p[0], reproj_p[1]]
        u = reproj_p[0]
        v = reproj_p[1]
        # dedxi -> partial to the camera pose
        jbBuffer[0, i] = idepth * dIdx
        jbBuffer[1, i] = idepth * dIdy
        jbBuffer[2, i] = -idepth * (u * dIdx + v * dIdy)
        jbBuffer[3, i] = -u * v * dIdx - (1 + v * v) * dIdy
        jbBuffer[4, i] = (1 + u * u) * dIdx + u * v * dIdy
        jbBuffer[5, i] = -v * dIdx + u * dIdy
    H = torch.matmul(jbBuffer.T, jbBuffer)
    b = torch.matmul(jbBuffer.T, e)
    loss = torch.sum(e)
    return H, b, loss
 ```