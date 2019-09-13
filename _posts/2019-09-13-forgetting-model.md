---
published: false
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

Here function \(\pi\) is the coefficient factor that controls how \(w\) will be updated.
