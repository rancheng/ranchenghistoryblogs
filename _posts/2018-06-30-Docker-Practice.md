---
layout: post
title: Docker Practice Notes
---
This post is about: How to remove docker images, run docker images and compose docker-compose file.

### Delete Docker images:


To delete all containers including its volumes use,

```sh
docker rm -vf $(docker ps -a -q)
```



To delete all the images,

```sh
docker rmi -f $(docker images -a -q)
```

Remember, you should remove all the containers before removing all the images from which those containers were created.



#### get into docker bash:

you have to run entrypoint before image:
```sh
docker run -it --entrypoint /bin/bash <image>
```
