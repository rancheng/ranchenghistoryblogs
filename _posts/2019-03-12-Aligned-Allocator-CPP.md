---
layout: post
title: Observations on Aligned Allocator
---

Allocators will normally span out a whole chunk of memory that's immediately available in heap, yet if you want to apply Intel SSE
optimization to seep up your computation pipeline, allocating space without alignment will introduce endless trouble.

We can solve any memory alignment problem with bit operations.

Here's a basic allocator template:
```cpp
template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((__intptr_t)(ptr+padT)) >> b) << b);
    return alignedPtr;
}
```

Don't worry, I'll explain line by line: 

`1 << b` in Line: `padT = 1 + ((1 << b)/sizeof(T))` 

is simply left shift b bits in the memory slot, which is equivalent to multiply $$2^b$$ by shifting left b bits in memory, 
and plus 1. `padT` will always move the pointer to the next slot that can be devisible by 4.

`new T[size + padT]` allocates the required size of memory plus the padding for alignment.

`(T*)(( ((__intptr_t)(ptr+padT)) >> b) << b)` is very confusing at first, but when you look into the core part: `>> b) << b`  
which is simply doing one thing: alignment:  

`(15 >> 4) << 4 = 0 (01111)`  
`(16 >> 4) << 4 = 16(10000)`  
`(19 >> 4) << 4 = 16(10011 -> 10000)`  
What's your observation?  
Anything that `mode 16 != 0` will be set as 0.

So, it's clear now that with the `padT` added to `ptr`, `ptr` will certainly be aligned into next memory block which address is divisible by 16.

Here's the full C++ code:
```cpp
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((__intptr_t)(ptr+padT)) >> b) << b);
    std::cout << " b \t\t\t\t\t\t\t\t\t\t\t" << b << std::endl;
    std::cout << " ptr \t\t\t\t\t\t\t\t\t\t" << ptr << std::endl;
    std::cout << " padT \t\t\t\t\t\t\t\t\t\t" << padT << std::endl;
    std::cout << "(T*)(__intptr_t)(ptr) \t\t\t\t\t\t" << (T*)(__intptr_t)(ptr) << std::endl;
    std::cout << "(T*)(__intptr_t)(ptr+padT) \t\t\t\t\t" << (T*)(__intptr_t)(ptr+padT) << std::endl;
    std::cout << "(T*)(__intptr_t)(ptr+padT) >> b\t\t\t\t" << (T*)(__intptr_t)(ptr+padT) << std::endl;
    std::cout << "(T*)(( ((__intptr_t)(ptr+padT)) >> b) << b)\t" << (T*)(( ((__intptr_t)(ptr+padT)) >> b) << b) << std::endl;
    std::cout<<" 1 "<< (T*)( ((__intptr_t)(ptr+padT)) >> b) << std::endl;
    std::cout<< " 2 " << (T*)(( ((__intptr_t)(ptr+padT)) >> b) << b) << std::endl;
    return alignedPtr;
}

int main() {
  int a[] = {1,2,3,4};
  std::vector<float*> ptrToDelete;
  float* idepth[8];
  float* b = allocAligned<4,float>(30, ptrToDelete);
  std::cout << " (19 >> 4) << 4 = " << (int)((19 >> 4) << 4) << std::endl;
  std::cout << (1 << 5) << std::endl;
  std::cout << "sizeof(float)" << sizeof(float) << std::endl;
  std::cout << "Hello World!\n" << b << std::endl;
}
```
and here's the terminal output (compiled with clang 7.0)
```sh
clang version 7.0.0-3~ubuntu0.18.04.1 (tags/RELEASE_700/final)
> clang++-7 -pthread -o main main.cpp
> ./main
 b                                          4
 ptr                                        0x10cde70
 padT                                       5
(T*)(__intptr_t)(ptr)                       0x10cde70
(T*)(__intptr_t)(ptr+padT)                  0x10cde84
(T*)(__intptr_t)(ptr+padT) >> b             0x10cde84
(T*)(( ((__intptr_t)(ptr+padT)) >> b) << b) 0x10cde80
 1 0x10cde8
 2 0x10cde80
 (19 >> 4) << 4 = 16
32
sizeof(float)4
Hello World!
0x10cde80
```
