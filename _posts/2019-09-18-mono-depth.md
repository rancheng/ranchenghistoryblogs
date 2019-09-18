---
published: false
---
Depth estimation has been a very hard task for monocular cameras, since the camera intrinsics varies a lot in different configurations, so it's very hard to migrate your trained model into another cheap sensors that with different configurations. Community has tried several different ways, multi-view stereo, virtual stereo, spatial propagation and even using the Bundle Adjustment and reprojection photometric methods to solve this issue in sequences of images. (Since static image doesn't have depth prior or the motion constrains, thus infeasible to estimate absolute pose and depth).

I have tried different monocular depth estimation methods: MonoDepth2, CSPN (2019 version, with affine model), and 
