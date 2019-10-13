---
layout: post
title: Solve BA with PyTorch
---

Since Bundle Adjustment is heavily depending on optimization backend, due to the large scale of Hessian matrix, solving Gauss-Newton directly is extremely challenging, especially when solving Hessian matrix and it's inverse. Scale of Hessian matrix is depending on number of map points, so to this end, we can first estimate the point inverse depths and then update the poses in two stage manner, however, this requires a very good initializaiton for depth estimation, and requires a convex local photometric loss to make sure positive definite property of the Hessian.
