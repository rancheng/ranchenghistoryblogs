---
title: old thoughts
layout: post
published: true
---
The match happened in one shot, which means the network in brain is a broadcast way, input was broadcasting towards each corner, and bouncing inside brain, eventually, those matched pattern will retain a high gain and activate the path.

To imagine this process, think about how a boy look at the cloud, in a blue sky, what is he thinking? He matched the cloud with any other clouds he saw in other days, but he continues looking at it, after a few seconds, different connections pops up, it likes a dog, likes an elephant, likes an ice-cream, like ...

Also, think about a flower, what's in your mind right now, everyone has a figure, but mine is a lotus, but when I think harder it's a daisy followed by rose, small white flower... All those experiments shows that memory has a delay on access all those memories, but they are connected (connected with flower in this case). and a delayed pop up means they have a travel distance, which means they are not in the same level on the tree (actually, it should be a graph). By building the graph, we connect related experience with different sensors, vision, scent， hearing, touch will grow their own graph each node represent a sensor result, and notice there's numerous combinations on the pattern, so it's nearly unlimited representation for each sensor. Human brain grows up with collect and grow the sensor nodes and the connections to other regions by daily life. Vision and scent and taste are connected by eating an orange. The strong feeling will give feed back to brain and broadcast back same way to different sensor nodes, thus create a unique memory.

What we are doing is maintaining the huge graph, the gate to navigate in different path build up the unique memory for each person. What Machine Learning and Deep Learning for now is building the bridge from sensory data into linguistic representation. Since linguistic data is multi-sensor fusion results, one world will trigger many sensor memories, which makes the single data source difficult to navigate to the right connections, what we can do is to introduce a whole new system of learning. Use the broadcasting mechanism to learn the world, which for simplicity, take MNIST as example, we are not using 32 kernels to filter out features of all 10 digits, instead, we can think about use the digits themselves to build the impressions.

$$ w = \sum^{T}_{t=0}{g(t, I_t)} $$

Where $$g(t, I_t)$$ is a diffusion function, which will diffuse out those unrelated part along time, and keep those key connections and convert into a topological graph.
