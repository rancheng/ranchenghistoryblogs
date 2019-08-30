---
layout: post
title: DSO Reviews and Future Extensions.
---

The biggest challenge of dso in this experiment is the low contrast problem (e.g.
white sand that have the same brightness everywhere) and the in-place rotation,
we have made several modification on the two occasions to make it more robust
for the underwater environment.

 - make prediction on inverse depth when the confidence of dso is low, we define the confidence as following:
 
 $$ C_i = \frac{(i-1) * E_{photo, i}}{\sum_0^{i-1}{E_{photo, j}}} $$
 
 global photometric error is large enough, and the active point hesssians drop under 30 percent of the last keyframe.
 Our strategy follows an assumption is that the robot follows a trajectory that has constant height over the sea floor, 
 thus if we detect a drop of confidence
 
 We have made several modification to the DSO algorithm to perform more robust and precise underwater. 
 One, we have applied the observation overlapping cancelling strategy over the low texture area
 (i.e. the sand area between coral reefs). By ignoring the samples come constantly from the same region, 
 we can average the distribution of our candidate points and get rid of the large estimation error over the $$\mathbf{SE(3)}$$. 
 In turn, when the candidate distribution concentrates, our candidate point hessians also drops, and this will trigger our 
 estimator to update the confidence of our pose. The confidence of estimation is defined as following:
 
 $$ C_{est} = \theta * \frac{\sum_0^{i-1}{E_j} * \mathbf{n_{ph,i}}}{E_i * (i-1) * max(\mathbf{n_{ph, 0...i-1})})} $$
 
 Here $$C_{est}$$ is the confidence of estimation, theta is a normalize factor, $$E_i$$ and $$E_j$$ are photometric error 
 for keyframe i and j, here $n_{ph,i}$ is the number of point hessians in keyframe i; Two, for scaling problem of the monocular
 version DSO, typically, the default idea is to make an arbitrary scale factor over each initialization.
 However, in our case, we have nearly constant distance over the sea floor, plus the distance change caused from current is 
 usually happens with a horizontal translation, thus can be captured by DSO after the initialization. 
 To simplify the scale problem, we introduced an empirical term over the scale factor to normalize the scale factor over 
 around 2-5 meters over the sea floor. This fixed normalizer will have positive influence over the estimation of the near 
 object and stitching the whole map; Three, the last practical problem is the motion blur due to the quick motion and a 
 sudden close up to the camera that roduce a large translation which fails DSO's reprojection process to track the new 
 point hessians. Our approach increases the max keyframe and samples more candidate point hessians whereas slightly 
 reduces the active point hessians, to make the competition more drastic and drop the outliers to further increase the 
 robustness of the estimations. When all above still fails, we will mandatory update all local point hessians and unify all 
 the inverse depths to the mode of all the depths. This can make sure our $$\mathbf{SE(3)}$$ resolver not blow up and keep a 
 stable estimation over the displacements; Four, we added an filter to the $$\mathbf{SE(3)}$$ which could help us annihilate 
 the drastic pitch and produce the folding of our map. The annihilation term was defined as following:
 
 $$ \omega_1  G_1 + \xi \omega_2 G_2 + \omega_3 G_3 \in \mathbf{SO(3)} $$
 
 $$ \xi = \frac{1}{1 + e^{-0.5 C_{est}}} $$
 
 We have bind our pitch estimation over the rotation in $$\mathbf{SO(3)}$$ with the confidence of how our estimation is, 
 if it's very low, we will lower our relative rotation $$G_2$$ term that corresponding to our pitch. 
 This is a little bit tricky in this experiment, but when we think about improvement the robustness of DSO, 
 we can try the similar ways, like keep tracking the confidence of the estimator and at the same time running a 
 low cost EKF to track the changes, if we lost the confidence, we can change immediately back to our EKF back-end 
 and run with our predictions, when the initialization was done, we can keep track on what we are doing rather 
 than lost from re-initialization.
 
