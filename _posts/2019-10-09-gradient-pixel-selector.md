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
 - calculate horizontal image gradient $$I_x$$, and vertical image gradient $$I_y$$, and get $$I_{xxyy} = ||I_x^2 + I_y^2||$$ as `absSquaredGradient` map (same size as original image). Then cut image into 32x32 small patches and count their gradients into histograms (HOG, 50 bins)
 - take first quantile of gradients (`hist[0]` is ~1024, take 90, which is top 10% highest gradients) as the HOG threshold of the scaled image (x32 smaller) and smooth the threshold map use it's mean value.
 - Finally, the vector `thsSmoothed` is the smoothed (x32 smaller) map to select the gradients. 
 
 Now we have the gradient selection threshold map, and the gradient map, next is to loop the image and select points according to the local gradients:


 
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
        Eigen::Vector3f const *const map0 = fh->dI;
        // 3 levels of sum suared gradients. 0 is the largest and 2 is the smallest
        float *mapmax0 = fh->absSquaredGrad[0];
        float *mapmax1 = fh->absSquaredGrad[1];
        float *mapmax2 = fh->absSquaredGrad[2];

        // why they don't define h1, h2???? do they default set the image as squared?????
        int w = wG[0];
        int w1 = wG[1];
        int w2 = wG[2];
        // only define hte first layer of height?
        int h = hG[0];


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

        // 4 different status, that makes the map_out 0 to 4 mapping of status pixel by pixel.
        // this is allocation of status write for map_out and pass back to the coarseInitializer
        // for now, I still don't know what those status used for, but will fill up later on
        //TODO: find out the function of PixelSelectorStatus
        // ##################
        // update: here there's 4 channels of status, which make the map_out or SelectionMap
        // a w*h*4 matrix.
        memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));


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
                                    }
                                // from this if all those following code are recording the map_out with those matched points
                                // and marked in different scales.
                                // map_out[some_index] = 1 means that this is the match point searched by the smallest scale.
                                if (bestIdx2 > 0) {
                                    map_out[bestIdx2] = 1;
                                    bestVal3 = 1e10;
                                    n2++; // this way it increased!!!
                                }
                            }

                        if (bestIdx3 > 0) {
                            map_out[bestIdx3] = 2; // mark the selected point in scale 2, which is even larger scale
                            bestVal4 = 1e10;
                            n3++;
                        }
                    }

                if (bestIdx4 > 0) {
                    map_out[bestIdx4] = 4; // this is the largest scale, which literally covers the whole image piexls.
                    n4++;
                }
            }
        // after finished the loop above, all the point that are suppose to be a match was selected and marked in map_out.
        // here map_out has 3 different scales. but every point marked in map_out are selected.
        // n3 n2 and n4 are the point size selected in different scales.
        return Eigen::Vector3i(n2, n3, n4);
    }
 ```
