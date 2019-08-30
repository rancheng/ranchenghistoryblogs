---
layout: post
title: Deploy Full sharelatex with Docker
---

An alternative way to use latex, host in local so that you can write paper without internet.

create a directory to save your repo.

```sh
cd ~
mkdir sharelatex-full
cd sharelatex-full
touch docker-compose.yml
```
and then write those contents to `docker-compose.yml`:

```
version: '2'
services:
  sharelatex:
      restart: always
      image: rigon/sharelatex-full:latest
      build: .
      container_name: sharelatex
      depends_on:
          - mongo
          - redis
      privileged: true
      ports:
          - 80:80
      network_mode: host
      links:
          - mongo
          - redis
      volumes:
          - ~/sharelatex_data:/var/lib/sharelatex
      environment:
          SHARELATEX_MONGO_URL: mongodb://localhost:27017/sharelatex
          SHARELATEX_REDIS_HOST: localhost
          SHARELATEX_APP_NAME: Our ShareLaTeX
  mongo:
      restart: always
      image: mongo
      container_name: mongo
      expose:
          - 27017
      network_mode: host
      volumes:
          - ~/mongo_data:/data/db

  redis:
      restart: always
      image: redis
      container_name: redis
      expose:
          - 6379
      network_mode: host
      volumes:
          - ~/redis_data:/data
```
Here we are using the host network to host `mongodb` and `redis`, and map the address back to `sharelatex` service.

----

Save docker-compose file and run the following command:
```sh
docker-compose up -d
```
Here `-d` means run in detached mode.

----

Head to `localhost/launchpad` in your web-browser, and create a new admin account, then start to use your sharelatex service.

`docker-compose` is more about a predefined script on running different components in the docker images, it pull those images from remote repo and run the services, and since it's in container, it won't break anything in your host system, this is a very nice container technique.
