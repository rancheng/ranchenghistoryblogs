---
published: false
---
if you installed cuda yet happen to be not able to find nvcc in your terminal, just navigate to the cuda libraries to execute the nvcc for checking the versions:

```sh
/usr/local/cuda/bin/nvcc --version
```

or create a link to `/usr/bin`:

```sh
ln -s /usr/local/cuda/bin/nvcc /usr/bin/nvcc
```