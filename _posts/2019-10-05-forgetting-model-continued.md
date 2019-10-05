---
layout: post
title: forgetting model continued
---

As in last post, I have proposed the way to slow down the conjugate gradient update on each dimension, instead, try to update part of the network can lead us faster convergence and more generalizability. Here's the proof.

Think about the gradient update on back propagation which is the error's partial to output, activation and weighted sum:

$$
\frac{\partial E}{\partial w} = \frac{\partial E}{\partial z} \frac{\partial E}{\partial h} \frac{\partial h}{\partial w}
$$

Which is simply chain rule. and here $$z$$ is the output of the current node, $$h$$ is the activation function, $$w$$ is the weight. Let's assume our network is using sigmoid activation function and here's our final result for updating a single weight:

$$
\Delta w = (o_i-t)o_n(1-o_i)o_{i-1}
$$

Here $$o$$ is the output and $$t$$ is the target (last layer), if this is not the last layer, the equation becomes a linear sum of the next layer's weight updates as the mimic of $$(o-t)$$.

$$
\Delta w_i = \sum{\Delta w_{i+1} * w_{i+1}} * (1 - o_i) * o_{i-1}
$$

To update each weight, we have to take those update sums in to consideration, think about that, what if those updates are mostly zero, thus the whole update will become zero (**vanishing gradient**), and this is not just linearly die out, it's exponentially. However, $$\Delta w$$ was based mostly on the gradient from the last layer and the input from previous layer, and they contribute bigger update if the input or the gradient is larger. Now we can treat this update as a function of gradients and inputs (hidden layer).

$$
\Delta w_i = g(\Delta w_{i+1}, o_{i-1})
$$

People has seeked uniformly way to regularize this step to prevent gradients from vanish or explode, the use of ReLU and jump connections are trying to avoid this step. And introduce a decay factor to prevent them from over-fitting. But why don't we try to maintain another coefficient matrix to control it's update speed? inspired by reinforcement learning, which we have a policy network guide up which directions to go, why can't we create such a matrix to guide where gradients should go? But how to build such a matrix? What property it should have?

Well, let's image the gradients are water and those weights are the pipes that water can pass. How to navigate those gradients? Our total water size is fixed, so water always go to the energy lower parts and release their gravity energy, thus, we can design a manifold similar to those pipes, or more intuitively, filters, each layer of network is regard as a filter, and the holes in it is not uniformly set, aka, they are not updating on the same pace. For those part of network which has already done a good job on prediction, we shall lock them down, on the contrary, we want make a bigger hole for those bad predicted parts to adjust their weights. This is another mimic for forgetting and remembering, those good weights with a very small update coefficient becomes the memories. And those flexible parts are very easy to be occupied by new samples come in and eventually becomes a forgetting network. By doing this, human can accumulately get the knowledge, whereas maintain a limited model size.

Now back to our formulars, how do we design this matrix? We can be sure that this matrix is about time and error: larger error contribute high forgetting, smaller error contribute high memorizing. And both forgetting and memorizing are decaying from time. Looks like a 太极 (Tai-chi). along the time, the forgetting and memorizing will be equalized. which means not memorizing nor forgetting.

Here we define our forgetting model as following:
$$
\pi(t, l, \Delta w) = e^{ - \frac{t}{\beta(\Delta w)l}}
$$

Here $$t$$ is time, $$l$$ is layer, and $$\Delta w$$ is the gradient for weight $$w$$.

and the $$\beta$$ function is defined as following:

$$
beta = =\begin{cases}
               beta, if \Delta w > th_b\\
               beta + 1, elsewise
         \end{cases}
$$

This $$\pi$$ function will give out the forgetting coeffient for each weight, since this is a function of $$\Delta w$$, thus will be regard as a constant for each update and will contribute to the gradient with the partials:

$$
\Delta w_i = \sum{\Delta w_{i+1} * w_{i+1} * \pi_{i+1}} * (1 - o_i) * o_{i-1}
$$

Here the $$\pi_{i+1}$$ is the forgetting term in the next layer.
