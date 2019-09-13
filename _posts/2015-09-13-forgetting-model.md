---
layout: post
title: forgetting model for MLP
---
instead of linearly sum up the input (weighted) from previous layer, we can introduce a novel layer that act as a receptor of the input signals, and adjust them accordingly, that way, it's easier to secure each gradient's direction and thus relief the pain of optimization which normally blindly navigating in the non-convex manifold.

![forgetting.png]({{site.baseurl}}/_posts/forgetting.png)

$$w = w + \frac{\partial E}{\partial w}$$

What if we don't want to jointly optimize all the weights, especially for those layers which are very close to the output, they are essentially more abstract filters, and changes over the top of hierarchical structure will trigger larger part search in parameter space. That is time consuming, and people do have several techniques to solve that issue by introducing **conjugate gradient** and dynamic capsules.

However this parameter manifold optimization has several fatal drawbacks: 
 - they are non-convex optimization, saddle points will ruin the whole pipeline
 - there's no fatigue or forgetting in this mechanism, which will result in catastrophical forgetting
 - though ReLU and more advanced Linear Unit family solved the gradient vanishing problem, the gradient propagation to the lower layers (layer that close to the input) are still degrading by magnitude which slow down the convergence.
 
So, by observing all the drawbacks above, I introduced a forgetting model in the multi-proceptron layers systems.

![forgetting2.png]({{site.baseurl}}/_posts/forgetting2.png)

Here function \(\pi\) is the coefficient factor that controls update rate for \(w\). Consider this learning system, which want to keep it's lowest energy and survive, they have to filter out those unrelated information and get the rewards to survive, similar to Reinforcement Learning hey? I learnt the trick from this community long time before until I understood the Q and policy update trick.

How to design such a system that seeking for low energy? Well, I don't know yet, but still get some hints on it.

Let's regard neural network a function approximator which are built from the error gradients aggregation. When the endless samples are adding to the system, the numerous samples are feeding to the system that they are stucking from one local minimal into another, and this parameter space manifold will be non-stationary, especially for the layers that close to the output part.

> consider a network that learns to auto-associate a large numberof patterns. The way in which the network ‘knows’ whether ornot it has seen a particular pattern before is by comparing thepattern on input and on output – the result of its having passedthrough  the  network.  If  there  is  very  little  difference,  it  con-cludes that it already ‘auto-associated’ that particular pattern. In another words, it had already seen it.  On the other hand, a largeinput–output  difference  means  that  it  has  encountered  a  new pattern.  But  now,  consider  what  happens  if  the  network  haslearned so many patterns that it has effectively learned the iden-tity function. Once the network can reliably produce on outputwhat it received on input for a large enough set of patterns, itwill  generalize  correctly  but  it  will  ‘remember’  virtually  any  pattern, whether or not it has actually ever seen it before. Thefundamental difficulty is that the network has then lost its abilityto  discriminate  previously  seen  input  from  new  input,  eventhough  it  is  generalizing  the  new  input  correctly.  Thus,  the  ability  to  generalize  to  the  identity  function  will  necessarilymean that there will be a loss of discrimination.The  problem  of  catastrophic  remembering  remains  an  im-portant one, and one for which current auto-associative connec-tionist memory models have no immediate answer.

Okay, now think about what will happen if all values of w are getting towards zeros, weights of connections are getting closed, that means that everything are forgetting, if \(w = 0\) that means everything is direct copy. But now you want to learn the limited parameter from infinite number of samples in real life and most of those samples are actually ambiguious and repeated, what will you do?

For what I can think about now is that through the course of learning, the learner (weights) are very flexible, but after a time of training, the learner will consolidate at least part of it's parameters to the converged local minimals, and perhaps eventually stayed at that local minimal.

I used to think that forgetting is to nudging all weight towards 0. But truth is not that case, at least not for now. High error gradients propagating back means that the weights are having a very high change of making errors, thus requires a very large step towards gradient direction, this is normally called forget, you forget means you are more versatile and flexible towards learning new stuff, right?

Now with that idea, we can further explore it to this concept:

 - high error \(\rightarrow\) trigger forgetting
 - low error \(\rightarrow\) trigger remebering

And if this process is constantly augmented through time, we can now define our final equation of function \(\pi\):

$$ \pi(t, l, \Delta w) = e^{- \frac{\beta(\Delta w) l}{t}}$$


