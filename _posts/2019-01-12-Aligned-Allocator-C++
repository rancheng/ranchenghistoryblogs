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
`1 << b` in Line: `padT = 1 + ((1 << b)/sizeof(T))` is simply left shift b bits in the memory slot, which is equivalent to multiply
$$2^b$$ by shifting left b bits in memory, and plus 1. `padT` will always move the pointer to the next slot that can be devisible
by 4.
