---
layout: post
title: gradient pixel selector in DSO
---

Most of the implementation details are prone to be neglected if you only read the paper, in this pose, I'll introduce the way DSO pick their candidate initialization anchor point hessians, and explain why the choose this stochastic gradient based initialization policy.

The core function in `PixelSelector2` is funciton `Eigen::Vector3i PixelSelector::select(const FrameHessian *const fh, float *map_out, int pot, float thFactor)`{:.cpp}. In function `select` the author emulate the convolution operation in different scale space. The original code seems to be very hard to understand due to C++ language feature, but when you look into it, and you will find most of the operations can be easily explained in vectorized operations in Python.

Now, let's start to explain this function line by line:

parameters: 
 - `FrameHessian* const fh` is the frame pointer, `const` constrained read-only property
 - `float* map_out` is actually a pointer to the selected point mask, which is the same size as original image
 - `int pot` is the potential field defined in the paper, here you can simply regard it as a local region for the pixel selector to search
 - `thFactor` is the threshold 
