
#pragma once

#include <thrust/execution_policy.h>

#include <visionaray/math/detail/math.h>

namespace visionaray
{

namespace detail
{

template <typename InputIt, typename Counts, typename Key>
__global__ void count_kernel(InputIt first, InputIt last, Counts counts, Key key)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    using T = typename std::iterator_traits<InputIt>::value_type;

    if (index < last - first)
    {
        atomicInc((unsigned*)&counts[key(first[index])], UINT_MAX);
    }
}

template <typename Counts>
__global__ void prefix_sum(Counts counts, int size)
{
    for (int i = 1; i < size; ++i)
    {
        counts[i] += counts[i - 1];
    }
}

template <typename InputIt, typename OutputIt, typename Counts, typename Key>
__global__ void scatter_kernel(InputIt first, InputIt last, OutputIt out, Counts counts, Key key)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < last - first)
    {
        auto addr = atomicDec((unsigned*)&counts[key(first[index])], UINT_MAX);
        atomicExch((unsigned*)&out[addr - 1], first[index]);
    }
}

} // detail

template <typename InputIt, typename OutputIt, typename Counts, typename Key>
void cuda_counting_sort(InputIt first, InputIt last, OutputIt out, Counts& counts, Key key = Key())
{
    int len = last - first;
    int block_size = 128;
    int grid_size = div_up(len, block_size);

    thrust::fill(
            thrust::device,
            counts.begin(),
            counts.end(),
            0
            );

    detail::count_kernel<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(first),
            thrust::raw_pointer_cast(last),
            thrust::raw_pointer_cast(counts.data()),
            key
            );


    detail::prefix_sum<<<1, 1>>>(
            thrust::raw_pointer_cast(counts.data()),
            counts.size()
            );


    Counts new_counts(counts);
    detail::scatter_kernel<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(first),
            thrust::raw_pointer_cast(last),
            thrust::raw_pointer_cast(out),
            thrust::raw_pointer_cast(new_counts.data()),
            key
            );
}

} // visionaray
