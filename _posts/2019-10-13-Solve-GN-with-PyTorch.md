---
layout: post
title: Solve BA with PyTorch
---

Since Bundle Adjustment is heavily depending on optimization backend, due to the large scale of Hessian matrix, solving Gauss-Newton directly is extremely challenging, especially when solving Hessian matrix and it's inverse. Scale of Hessian matrix is depending on number of map points, so to this end, we can first estimate the point inverse depths and then update the poses in two stage manner, however, this requires a very good initializaiton for depth estimation, and requires a convex local photometric loss to make sure positive definite property of the Hessian.

![gn_opt.png]({{site.baseurl}}/images/gn_opt.png)

Figure above illustrated Gauss-Newton method pipeline, which is iteratively estimating $$\Delta x$$ and updating $$x$$ to the convergence point where $$\Delta x$$ close to zero. This pipeline can be well modeled by `PyTorch` where we can easily use `autograd` to solve Jacobians and pseudo-inverse of Hessian matrix.

![ba_nn.png]({{site.baseurl}}/images/ba_nn.png)

```python
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
```
