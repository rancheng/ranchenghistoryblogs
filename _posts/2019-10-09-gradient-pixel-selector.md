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
 - `thFactor` is the threshold scale factor (2 in paper) to scale up the smoothed threshold filtered histogram.
 
If you are confused by the thresholding technique in the code, here's the simplified explanation:
 - calculate horizontal image gradient $$I_x$$, and vertical image gradient $$I_y$$, and get $$I_x^2 + I_y^2$$ as `absSquaredGradient` map (same size as original image). Then cut image into 32x32 small patches and count their gradients into histograms (HOG, 50 bins)
 - take first quantile of gradients (`hist[0]` is ~1024, take 90, which is top 10% highest gradients) as the HOG threshold of the scaled image (x32 smaller) and smooth the threshold map use it's mean value.
 - Finally, the vector `thsSmoothed` is the smoothed (x32 smaller) map to guide whether a pixel point should be select or not according to it's gradient. 
 
 Now we have the gradient selection threshold map, and the gradient map, next is to loop the image and select points according to the local gradients:

![pixel_selector.png]({{site.baseurl}}/images/pixel_selector.png)
 
 This is the original code with my comments:
 ```cpp
// this function loops through different scale space and for each space, they select the point with
// dx and dy collinear with the random direction.
Eigen::Vector3i PixelSelector::select(const FrameHessian *const fh,
                                      float *map_out, int pot, float thFactor) {
    // int pot is potential, defined in the class, but what is potential defined in paper?
    // my guess is pot here is a scale factor for how much search space they want to cover
    // in each scale space.
    // the map0 now is used to define image.... notation abuse, understood.
```
`mapmax0` ... are linked to first 3 level of `absSquaredGrad`, note that `absSquaredGrad` has 6 `PYR_LEVELS` levels, and the `mapmax` variables are used only on the first 3 smallest patches, (stepsize are x1: pixelwise, x2: 4x pixel wise, x3: 16x pixelwise). So this correspondence make sure you can find the gradient accordingly.

```cpp
Eigen::Vector3f const *const map0 = fh->dI;
// 3 levels of sum suared gradients. 0 is the largest and 2 is the smallest
float *mapmax0 = fh->absSquaredGrad[0];
float *mapmax1 = fh->absSquaredGrad[1];
float *mapmax2 = fh->absSquaredGrad[2];
```
Nothing to explain here, this is just record the width and height in different scales.

```cpp
// why they don't define h1, h2???? do they default set the image as squared?????
// because they reshaped the matrix into 1d, so only width is needed
int w = wG[0];
int w1 = wG[1];
int w2 = wG[2];
// only define hte first layer of height?
int h = hG[0];
```
Predefined direction vector, which is basically record 16 directions in a circle origin from (0, 0).

```cpp
const Vec2f directions[16] = {
        Vec2f(0, 1.0000),
        Vec2f(0.3827, 0.9239),
        Vec2f(0.1951, 0.9808),
        Vec2f(0.9239, 0.3827),
        Vec2f(0.7071, 0.7071),
        Vec2f(0.3827, -0.9239),
        Vec2f(0.8315, 0.5556),
        Vec2f(0.8315, -0.5556),
        Vec2f(0.5556, -0.8315),
        Vec2f(0.9808, 0.1951),
        Vec2f(0.9239, -0.3827),
        Vec2f(0.7071, -0.7071),
        Vec2f(0.5556, 0.8315),
        Vec2f(0.9808, -0.1951),
        Vec2f(1.0000, 0.0000),
        Vec2f(0.1951, -0.9808)};
```
Clear out `map_out`, it's the mask matrix for pixel selection.

```cpp
// 4 different status, that makes the map_out 0 to 4 mapping of status pixel by pixel.
// this is allocation of status write for map_out and pass back to the coarseInitializer
// for now, I still don't know what those status used for, but will fill up later on
//TODO: find out the function of PixelSelectorStatus
// ##################
// update: here there's 4 channels of status, which make the map_out or SelectionMap
// a w*h*4 matrix.
memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));
```
`dw1` and `dw2` are down weight scale factors. Used to down weight the `pixelTH0` and `pxielTH1`, they are the smoothed pixel-wise threshold. `n2`, `n3`, `n4` are the selected point numbers in different scales.

```cpp
// down weighting constant, don't know what they exactly doing.
float dw1 = setting_gradDownweightPerLevel;
float dw2 = dw1 * dw1;
//------------------------------------Explain of loop:----------------------------------------------

// this loop is searching in different scales.

// first scale y4, x4 which is looping trough the whole image space

// second scale y3, x3 which looping through the 4*pot size small patch start at y4, x4.

// now on y2, x2 and y1, x1 should be even smaller size of patch to search.

// this is equivalent to the convolution through different scale space.

// and they change directions in each loop step in each scale space.

// eventually, in the loop body, they capture those largest dirNorm (which means gradient vector are collinear

// with the random direction)

// intuitively, we can regard all point like that are matched point and considered as selected point.

//----------------------------------------------------------------------------------------
// n3 n2 and n4 are index of vector returned. increased at each for loop.
int n3 = 0, n2 = 0, n4 = 0;
```
Start of first loop, which is cutting the original image into 32x32 small patches, and the upcoming nested loops are searching inside each patch. Since `y4 += (4 * pot)`, the first loop stride step is `4*pot`, since `pot` will change on function `makeMaps` in each recursion, this sliding operation is dynamic.

```cpp
// pot here is the potential defined in the function or passed from coarse initializer
// here I understand as the stride.
// !Notice: this starts from y4 and x4 loop.
for (int y4 = 0; y4 < h; y4 += (4 * pot))
    for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
        // 4*pot, why it's 4*pot? h-y4?
        int my3 = std::min((4 * pot), h - y4);
        // w-x4? so select the min value of 4*pot or w-x4, this x index will always be 4*pot when it reach to the end of
        // the matrix, and why they will only get the last few steps? here, 4*pot under setting, is 12, so they will
        // always be 12 until x4 reaches end of w. but x4 will jump each 4*pot for each iteration.
        int mx3 = std::min((4 * pot), w - x4);
        int bestIdx4 = -1;
        float bestVal4 = 0;
        // 0xF is 15, and randomPattern[n2] is randomPattern[0] for now, so this is to sample the same direction.
        // shouldn't this be n4? randomPattern[n2] ?
        // randomPattern is generated on the constructor.
        Vec2f dir4 = directions[randomPattern[n2] & 0xF];
        // !Notice: this is y3 and x3 loop.
```
Nested loop 1, this is the green square in the left part of diagram above. It start at `x4`, `y4` basis and loop around a `mx3` by `my3` sized patch, which is `4*pot` in center and `h-y4` or `w-x4` in the boarder. In the end, they choose a random direction `dir3`. `& 0xF` means choose the lower 4 bits as effective values, `0xF` is `...0000000000001111`, the `...` means all 0, by `&` operator, the bitwise operation will filter out all the higher parts to 0 and only keep the lower 4 bits effective, which in turn can represent `[0, 15]` range.
 
```cpp
for (int y3 = 0; y3 < my3; y3 += (2 * pot))
    for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
        int x34 = x3 + x4;
        int y34 = y3 + y4;
        int my2 = std::min((2 * pot), h - y34);
        int mx2 = std::min((2 * pot), w - x34);
        int bestIdx3 = -1;
        float bestVal3 = 0;
        // shouldn't this be n3?
        Vec2f dir3 = directions[randomPattern[n2] & 0xF];
```
Same thing in this loop, start from the y3 basis, and loop with `pot` size stride

```cpp
for (int y2 = 0; y2 < my2; y2 += pot)
    for (int x2 = 0; x2 < mx2; x2 += pot) {
        int x234 = x2 + x34;
        int y234 = y2 + y34;
        int my1 = std::min(pot, h - y234);
        int mx1 = std::min(pot, w - x234);
        int bestIdx2 = -1;
        // so this controls the loop for the update on n2..
        // n2 is the key looper to choose direction, so that means if
        // you didn't find any gradient that has higher direction norm
        float bestVal2 = 0;
        // this should be n2.
        Vec2f dir2 = directions[randomPattern[n2] & 0xF];
```
In this loop, note that the stride becomes 1, that means this is the finally in pixel-wise level selection, and thus the kernel of this function. first part is simple, likewise, they defined the starting positions, filter out out of boundary points.

```cpp
// seems like they are searching around different scales, now comes to the 1 stride
for (int y1 = 0; y1 < my1; y1 += 1)
    for (int x1 = 0; x1 < mx1; x1 += 1) {
        assert(x1 + x234 < w);
        assert(y1 + y234 < h);
        // loop the small patch in different big strides, now I understand they are searching in different
        // patches are small when they reach to the end of the strides.
        int idx = x1 + x234 + w * (y1 + y234); // x234 = x2 + x3 + x4... same as y234, this is just offsets.
        int xf = x1 + x234;
        int yf = y1 + y234;

        if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) continue;
```
This part they find out the threshold for each pixel in x32 smaller scale, and record them into `pixelTH0`, notice this is happened in each pixel in the small neighbourhood patch, the pixel threshold is actually the threshold to determine whether a pixel should be selected or not by comparing it's gradients with this threshold.

```cpp
// this is the pixels index, why the hell they will down weight those index??????
// beacause xf>>5 = xf/2^5 = xf/32, which is the smallest scale
// this is indexing thsSmoothed[x32, y32]
float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep]; // thsStep is w32
// down weight those index will shrink those points in the threshold.
// multiply dw1 and dw1 they lifted the bar of the threshold.
float pixelTH1 = pixelTH0 * dw1;
float pixelTH2 = pixelTH1 * dw2;

// this is the single abs gradient in the local index.
// thFactor now is 2...
float ag0 = mapmax0[idx];
// this pixelTH0 is just the min threshold for the gradient
// in order to find out more valid gradient, they use histogram to store all
// the gradients, and normalize throughout the down sampled image
// now they just want to use this threshold to pick up point according to gradient
// now I understand why they name it as pixel selector...
```
See, here's the comparison, the `thFactor` is 2 in the paper, and ag0 is the gradient, if the gradient is above threshold to choose, they will find the norm of the gradient and the random direction choosed above, then loop out the point with the highest direction norm. That controls the points gradients to be as random as possible, to make sure the later on reprojection loss function in 8 directions sums up to be convex locally.

```cpp
if (ag0 > pixelTH0 * thFactor) {
    // this will give the last two scales of abs gradient
    // remember this map0 is the dI, dI is 3 channels, color, dx, dy,
    // now tail<2> selects dx and dy channel.
    // this explained why they use ag0d -> d means gradient.
    // ag absolute gradient? 0 represent scale 0 which is the original image.
    Vec2f ag0d = map0[idx].tail<2>();
    // ag0d.dot(dir2) will give the direction norm? dot product will be
    // zero if ag0d is perpendicular to dir2...
    // dir2 is the random direction sampled by n2...
    // that means n2 will change if dir2 is not perpendicular to ag0d.
    // which means they are finding all the non-rthonormal basis...
    // and eventually converge to the minimal angle and until they are
    // in the same direction...
    float dirNorm = fabsf((float) (ag0d.dot(dir2)));
    if (!setting_selectDirectionDistribution) dirNorm = ag0;
    // bestIdx2,3,4 are used to update the n2 n3 and n4
    // if it's not orthogonal, then update bestVal2...
    // and if it's less angle, the higher the dirNorm
    // this step is to align the direction to gradient angle...
    if (dirNorm > bestVal2) {
        bestVal2 = dirNorm;
        bestIdx2 = idx;
        bestIdx3 = -2;
        bestIdx4 = -2;
    }
}
```
In this part, same logic, but in different scale, which is in larger patch, firstly we choose pixelwise in smallest patch, now we choose the best point in the neighbourhood: `(int) (xf * 0.5f + 0.25f) + (int) (yf * 0.5f + 0.25f) * w1` this choose the middle point in the 2x2 patch.

```cpp
// this continue will jump the loop
// because they found the best alignment on this loop,
// move to another x1...
if (bestIdx3 == -2) continue;
// same as above, this is for the second scale space. which
// is w/2, h/2 size gradient image.
// again, will find the most aligned direction along image gradient.
// here why on mapmax1 they shrink size of x and y
// because mapmax1 itself is the smaller sized map. width height is w/2 h/2
// + 0.25 is to compensate for what? still unknown, will investigate later...
float ag1 = mapmax1[(int) (xf * 0.5f + 0.25f) + (int) (yf * 0.5f + 0.25f) * w1];
if (ag1 > pixelTH1 * thFactor) {
    Vec2f ag0d = map0[idx].tail<2>();
    float dirNorm = fabsf((float) (ag0d.dot(dir3)));
    if (!setting_selectDirectionDistribution) dirNorm = ag1;

    if (dirNorm > bestVal3) {
        bestVal3 = dirNorm;
        bestIdx3 = idx;
        bestIdx4 = -2;
    }
}
```
Aggregate the gradient from even larger patch, 4x4 scale (`(int) (xf * 0.25f + 0.125) + (int) (yf * 0.25f + 0.125) * w2`) and choose the largest point in that patch, name as `bestVal4` and `bestIdx4`, this `bestIdx` will be write out to the selection mask (`map_out`) later on.

```cpp
if (bestIdx4 == -2) continue;

float ag2 = mapmax2[(int) (xf * 0.25f + 0.125) +
                    (int) (yf * 0.25f + 0.125) * w2];
if (ag2 > pixelTH2 * thFactor) {
    Vec2f ag0d = map0[idx].tail<2>();
    float dirNorm = fabsf((float) (ag0d.dot(dir4)));
    if (!setting_selectDirectionDistribution) dirNorm = ag2;

    if (dirNorm > bestVal4) {
        bestVal4 = dirNorm;
        bestIdx4 = idx;
    }
}
```
you can see that they write the `bestIdx2` into `map_out` and marked as 1, which means they are selected in the pixelswise patch. `n2` counts the selected point size in this cale

```cpp
// from this if all those following code are recording the map_out with those matched points
// and marked in different scales.
// map_out[some_index] = 1 means that this is the match point searched by the smallest scale.
if (bestIdx2 > 0) {
    map_out[bestIdx2] = 1;
    bestVal3 = 1e10;
    n2++; // this way it increased!!!
}
```
Here's the scale 2x2 patch, and count the points selected.

```cpp
if (bestIdx3 > 0) {
    map_out[bestIdx3] = 2; // mark the selected point in scale 2, which is even larger scale
    bestVal4 = 1e10;
    n3++;
}
```
4x4 patch.
```cpp
if (bestIdx4 > 0) {
    map_out[bestIdx4] = 4; // this is the largest scale, which literally covers the whole image piexls.
    n4++;
}
```
After write out the selection mask `map_out` then return the point numbers selected in each scale.
```cpp
// after finished the loop above, all the point that are suppose to be a match was selected and marked in map_out.
// here map_out has 3 different scales. but every point marked in map_out are selected.
// n3 n2 and n4 are the point size selected in different scales.
return Eigen::Vector3i(n2, n3, n4);
 ```

This whole function searches the random directional points with local gradients above certain threshold. Since it's invoked in `makeMaps` function and this function is a recursive function, the potential search area is changing according to each recursion, I will help you figure out what exactly is going on in the `makeMaps` recursive point selection operations. 

In order to have a better understanding on the point selection policy, let's implement the whole pipeline in python ([source code](https://github.com/rancheng/deep_mono_vo/blob/master/pixelSelector.py)) and run the experiments on Kitti dataset. Final result is as following:

![dso_psel.png]({{site.baseurl}}/images/dso_psel.png)

From the figure above, we can observe that the selection policy by DSO governed the local region's density thus make the candidate reprojection points evenly spread in the whole image, whereas the laplacian of gaussian point selector are concentrated around the edges. However, despite DSO's sample strategy spread the sample point as even as possible, they are still depend on the local image gradients. This is reasonable, in order to calculate the photometric error and construct the local BA problem as strong convex as possible, gradient based sample technique is the best choice by far.
