// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_EX_WAVEFRONT_PATHTRACER_PATHTRACER_H
#define VSNRAY_EX_WAVEFRONT_PATHTRACER_PATHTRACER_H 1

#define BENCHMARK
//#define SORT_BY_MATERIAL_TYPE

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <new>
#include <numeric>
#include <tuple>
#include <utility>


#ifdef __CUDACC__
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#endif

#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/camera.h>
#include <visionaray/get_surface.h>
#include <visionaray/random_sampler.h>
#include <visionaray/tags.h>
#include <visionaray/traverse.h>

#include <common/timer.h>

#include "integer_sequence.h"
#include "parallel_for.h"

#ifdef __CUDACC__
#include "cuda_counting_sort.h"
#endif

#ifndef __CUDACC__
#define __device__
#endif

#define MATERIAL_MAX 4

#define TRY_ALLOC(expression)                                                   \
try                                                                             \
{                                                                               \
    expression;                                                                 \
}                                                                               \
catch (std::bad_alloc&)                                                         \
{                                                                               \
    std::cerr << "Memory allocation failed in line: " << __LINE__ << '\n';      \
    exit(EXIT_FAILURE);                                                         \
}

template <typename Tuple, typename Func, size_t ...Indices>
void for_each_impl(Tuple&& tuple, Func&& func, visionaray::index_sequence<Indices...>)
{
    int ignore[] = {
        1,
        (func(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...
        };
    (void)ignore;
}

template <typename Tuple, typename F>
void for_each(Tuple&& tuple, F&& f) {
    constexpr std::size_t N = std::tuple_size<typename std::remove_reference<Tuple>::type>::value;
    for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                  visionaray::make_index_sequence<N>{});
}


namespace visionaray
{

class pathtracer
{
public:

    struct shade_next_t
    {
        shade_next_t(pathtracer& parent, unsigned* counts, unsigned bits)
            : parent(parent)
            , counts(counts)
            , bits(bits)
            , first(0)
            , last(0)
            , counter(0)
        {
        }

        template <typename Params>
        void operator()(Params par)
        {
            first = last;
            last = counts[counter];
            parent.shade(
                    parent.pointer_cast(parent.rays.data()),
                    parent.pointer_cast(parent.hit_recs.data()),
                    parent.pointer_cast(parent.indices.data()) + first,
                    parent.pointer_cast(parent.indices.data()) + last,
                    par,
                    parent.pointer_cast(parent.throughputs.data()),
                    bits
#if defined(__CUDACC__) && defined(SORT_BY_MATERIAL_TYPE)
                  , counter
#endif
                    );
            bits >>= 1;
            counter++;
        }

        pathtracer& parent;
        unsigned* counts;
        unsigned bits;
        int first;
        int last;
        int counter;
    };

private:

#ifdef BENCHMARK
    std::vector<double> intersecting;
    std::vector<double> sorting;
    std::vector<double> shading;
    std::vector<double> compaction;
    std::vector<double> blending;
#endif

#ifdef __CUDACC__
    using bvh_ref = cuda_index_bvh<model::triangle_type>::bvh_ref;
    using R = basic_ray<float>;
    using HR = decltype(intersect(std::declval<R>(), std::declval<bvh_ref>()));

    thrust::device_vector<R> rays;
    thrust::device_vector<HR> hit_recs;
    thrust::device_vector<int> indices;
    thrust::device_vector<int> indices_back;
    thrust::device_vector<spectrum<float>> throughputs;

    template <typename T>
    inline auto pointer_cast(T pointer)
        -> decltype(thrust::raw_pointer_cast(pointer))
    {
        return thrust::raw_pointer_cast(pointer);
    }


#else
    using bvh_ref = index_bvh<model::triangle_type>::bvh_ref;
    using R = basic_ray<float>;
    using HR = decltype(intersect(std::declval<R>(), std::declval<bvh_ref>()));

    aligned_vector<R> rays;
    aligned_vector<HR> hit_recs;
    aligned_vector<int> indices;
    aligned_vector<int> indices_back;
    aligned_vector<spectrum<float>> throughputs;

    template <typename T>
    inline T* pointer_cast(T* pointer)
    {
        return pointer;
    }
#endif

public:

    template <typename Params, typename RT>
    void frame(Params params_pack, RT& rt, camera const& cam, unsigned& frame_num)
    {
#ifdef BENCHMARK
        if (frame_num == 0)
        {
            intersecting.clear();
            sorting.clear();
            shading.clear();
            compaction.clear();
            blending.clear();
        }
#endif

        rt.begin_frame();

#if defined(SORT_BY_MATERIAL_TYPE)
        auto p = std::get<0>(params_pack);
        auto& params = p;
#else
        Params& params = params_pack;
#endif

        pathtracer::make_primary_rays(rays.begin(), cam, rt.width(), rt.height());
#ifdef __CUDACC__
        thrust::sequence(thrust::device, indices.begin(), indices.end(), 0, 1);
        thrust::fill(thrust::device, hit_recs.begin(), hit_recs.end(), HR());
        thrust::fill(thrust::device, throughputs.begin(), throughputs.end(), spectrum<float>(1.0));
#else
        std::iota(indices.begin(), indices.end(), 0);
        std::fill(hit_recs.begin(), hit_recs.end(), HR());
        std::fill(throughputs.begin(), throughputs.end(), spectrum<float>(1.0));
#endif

        size_t compact = indices.size();

#ifdef __CUDACC__
        // NOTE: CUDA event-based timing causes kernel execution
        // in separate streams to run sequentially. In order to
        // measure shade throughput, we therefore lazily use a
        // simple timer instead of the cuda::timer and activate
        // the call to cudaDeviceSynchronize() below!
        //cuda::timer t;
        timer t;
#else
        timer t;
#endif

        size_t num_rays = 0;

        for (unsigned bounce = 0; bounce < params.num_bounces; ++bounce)
        {
            num_rays += compact;

            pathtracer::primitive_intersect(
                    pointer_cast(rays.data()),
                    pointer_cast(hit_recs.data()),
                    pointer_cast(indices.data()),
                    pointer_cast(indices.data()) + compact,
                    params.prims.begin,
                    params.prims.end
                    );

#ifdef BENCHMARK
            //cudaDeviceSynchronize();
            intersecting.push_back(t.elapsed());
            t.reset();
#endif

#if defined(SORT_BY_MATERIAL_TYPE)

#ifdef __CUDACC__
            auto hit_recs_ptr = thrust::raw_pointer_cast(hit_recs.data());

            thrust::device_vector<unsigned> d_counts(MATERIAL_MAX + 1);
            cuda_counting_sort(
                    pointer_cast(indices.data()),
                    pointer_cast(indices.data()) + compact,
                    pointer_cast(indices_back.data()),
                    d_counts,
                    [=] __device__ (int index)
                    {
                        unsigned bits = 0x80000000;
                        for (int i = 0; i < MATERIAL_MAX; ++i)
                        {
                            if (hit_recs_ptr[index].hit && hit_recs_ptr[index].geom_id & bits)
                            {
                                return i;
                            }

                            bits >>= 1;
                        }

                        //assert(!hit_recs_ptr[index].hit);
                        return MATERIAL_MAX;
                    }
                    );

            thrust::copy(
                    thrust::device,
                    indices_back.begin(),
                    indices_back.begin() + compact,
                    indices.begin()
                    );

            thrust::host_vector<unsigned> counts(d_counts);
#else

            std::array<unsigned, MATERIAL_MAX + 1> counts;
            paralgo::counting_sort(
                    pointer_cast(indices.data()),
                    pointer_cast(indices.data()) + compact,
                    pointer_cast(indices_back.data()),
                    counts,
                    [=] __device__ (int index)
                    {
                        unsigned bits = 0x80000000;
                        for (int i = 0; i < MATERIAL_MAX; ++i)
                        {
                            if (hit_recs[index].hit && hit_recs[index].geom_id & bits)
                            {
                                return i;
                            }

                            bits >>= 1;
                        }

                        assert(!hit_recs[index].hit);
                        return MATERIAL_MAX;
                    }
                    );

            std::copy(
                    indices_back.begin(),
                    indices_back.begin() + compact,
                    indices.begin()
                    );
#endif

#ifdef BENCHMARK
            //cudaDeviceSynchronize();
            sorting.push_back(t.elapsed());
            t.reset();
#endif

            unsigned bits = 0x80000000;

            shade_next_t shade_next(*this, counts.data(), bits);
            for_each(params_pack, shade_next);

            // Handle paths that just exited
            pathtracer::shade(
                    pointer_cast(rays.data()),
                    pointer_cast(hit_recs.data()),
                    pointer_cast(indices.data()) + counts[MATERIAL_MAX - 1],
                    pointer_cast(indices.data()) + counts[MATERIAL_MAX],
                    params,
                    pointer_cast(throughputs.data()),
                    0x0
#if defined(__CUDACC__) && defined(SORT_BY_MATERIAL_TYPE)
                  , MATERIAL_MAX
#endif
                    );
#else
            pathtracer::shade(
                    pointer_cast(rays.data()),
                    pointer_cast(hit_recs.data()),
                    pointer_cast(indices.data()),
                    pointer_cast(indices.data()) + compact,
                    params,
                    pointer_cast(throughputs.data()),
                    0x0
                    );

#endif

#ifdef BENCHMARK
            // Call this to measure CUDA shade() throughput!
            //cudaDeviceSynchronize();
            shading.push_back(t.elapsed());
            t.reset();
#endif

            auto it = pathtracer::compact(
                    pointer_cast(indices.data()),
                    pointer_cast(indices.data()) + compact,
                    pointer_cast(indices_back.data())
                    );

            compact = it - pointer_cast(indices_back.data());

            swap_index_buffers();

#ifdef BENCHMARK
            //cudaDeviceSynchronize();
            compaction.push_back(t.elapsed());
            t.reset();
#endif
        }

        pathtracer::terminate_active(
                pointer_cast(indices.data()),
                pointer_cast(indices.data()) + compact,
                pointer_cast(throughputs.data())
                );

        pathtracer::blend(
                pointer_cast(indices.data()),
                pointer_cast(indices.data()) + indices.size(),
                pointer_cast(throughputs.data()),
                rt.ref(),
                ++frame_num
                );

#ifdef BENCHMARK
        //cudaDeviceSynchronize();
        blending.push_back(t.elapsed());
        t.reset();
#endif

        rt.end_frame();

#ifdef BENCHMARK
        if (frame_num % 20 == 0)
        {
            double in = std::accumulate(intersecting.begin(), intersecting.end(), 0.0) / intersecting.size();
            double so = std::accumulate(sorting.begin(), sorting.end(), 0.0) / sorting.size();
            double sh = std::accumulate(shading.begin(), shading.end(), 0.0) / shading.size();
            double co = std::accumulate(compaction.begin(), compaction.end(), 0.0) / compaction.size();
            double bl = std::accumulate(blending.begin(), blending.end(), 0.0) / blending.size();

            std::cout << "Rays:         " << num_rays << std::endl;
            std::cout << "Intersection: " << in << ", Mrays/s: " << ((num_rays / 1000000.0) / in) / params.num_bounces << std::endl;
            std::cout << "Sorting     : " << so << ", Mrays/s: " << ((num_rays / 1000000.0) / so) / params.num_bounces << std::endl;
            std::cout << "Shading     : " << sh << ", Mrays/s: " << ((num_rays / 1000000.0) / sh) / params.num_bounces << std::endl;
            std::cout << "Compaction  : " << co << ", Mrays/s: " << ((num_rays / 1000000.0) / co) / params.num_bounces << std::endl;
            std::cout << "Blending    : " << bl << ", Mrays/s: " << ((num_rays / 1000000.0) / bl) << std::endl;

//            intersecting.clear();
//            intersecting.shrink_to_fit();
//            sorting.clear();
//            sorting.shrink_to_fit();
//            shading.clear();
//            shading.shrink_to_fit();
//            compaction.clear();
//            compaction.shrink_to_fit();
//            blending.clear();
//            blending.shrink_to_fit();
        }
#endif
    }

    void resize(int w, int h)
    {
        TRY_ALLOC(rays.resize(w * h));
        TRY_ALLOC(hit_recs.resize(w * h));
        TRY_ALLOC(indices.resize(w * h));
        TRY_ALLOC(indices_back.resize(w * h));
        TRY_ALLOC(throughputs.resize(w * h));
    }

    template <typename Rays>
    inline void make_primary_rays(Rays rays, camera const& cam, int width, int height)
    {
        using S = typename R::scalar_type;

        //  front, side, and up vectors form an orthonormal basis
        auto f = normalize( cam.eye() - cam.center() );
        auto s = normalize( cross(cam.up(), f) );
        auto u =            cross(f, s);

        vec3 eye   = cam.eye();
        vec3 cam_u = s * tan(cam.fovy() / 2.0f) * cam.aspect();
        vec3 cam_v = u * tan(cam.fovy() / 2.0f);
        vec3 cam_w = -f;

        parallel_for(
            random_sampler<S>{},
            0, width * height,
            [=] __device__ (int index, random_sampler<S>& samp)
            {
                int x = index % width;
                int y = index / width;
                rays[y * width + x] = detail::make_primary_rays(
                        R{},
                        pixel_sampler::jittered_blend_type{},
                        samp,
                        x,
                        y,
                        width,
                        height,
                        eye,
                        cam_u,
                        cam_v,
                        cam_w
                        );
            }
            );
    }

    template <typename Rays, typename HitRecords, typename IndexIt, typename PrimIt>
    inline void primitive_intersect(
            Rays        rays,
            HitRecords  hit_recs,
            IndexIt     indices_first,
            IndexIt     indices_last,
            PrimIt      prims_first,
            PrimIt      prims_last
            )
    {
        parallel_for(0, indices_last - indices_first,
            [=] __device__ (int index)
            {
                auto it = indices_first + index;

                if (*it >= 0)
                {

                    // FIXME (in library)
#ifdef __CUDACC__
                    HR result;
                    for (auto p = prims_first; p != prims_last; ++p)
                    {
                        auto hr = intersect(rays[*it], *p);
                        update_if(result, hr, hr.hit && hr.t < result.t);
                    }
                    hit_recs[*it] = result;
#else
                    hit_recs[*it] = closest_hit(rays[*it], prims_first, prims_last);
#endif
                }
            }
            );
    }

    template <typename Rays, typename HitRecords, typename IndexIt, typename Params, typename Is>
    inline void shade(
            Rays        rays,
            HitRecords  hit_recs,
            IndexIt     indices_first,
            IndexIt     indices_last,
            Params      params,
            Is          throughputs,
            unsigned    material_bits
#if defined(__CUDACC__) && defined(SORT_BY_MATERIAL_TYPE)
          , int stream
#endif
            )
    {
        using R  = typename std::iterator_traits<Rays>::value_type;
        using S  = typename R::scalar_type;
        using V  = vector<3, S>;
        using HR = typename std::iterator_traits<HitRecords>::value_type;
        using C  = typename std::iterator_traits<Is>::value_type;

        parallel_for(
            random_sampler<S>{},
            0, indices_last - indices_first,
            [=] __device__ (int index, random_sampler<S>& samp)
            {
                auto it = indices_first + index;

                if (*it < 0)
                {
                    return;
                }

                R& ray = rays[*it];
                HR hit_rec = hit_recs[*it];
                auto& dst = throughputs[*it];

                // Handle rays that just exited (TODO: SIMD)
                if (!hit_rec.hit)
                {
                    dst *= C(from_rgba(params.ambient_color));
                    *it = ~(*it);
                    return;
                }

                V refl_dir;
                V view_dir = -ray.dir;

#if defined(SORT_BY_MATERIAL_TYPE)
                hit_rec.geom_id = hit_rec.geom_id & ~material_bits;
#endif
                auto surf = get_surface(hit_rec, params);

                auto n = surf.shading_normal;

#if 1 // two-sided
                n = faceforward( n, view_dir, surf.geometric_normal );
#endif

                S pdf(0.0);
                auto sr     = make_shade_record<Params, S>();
                sr.active   = hit_rec.hit;
                sr.normal   = n;
                sr.view_dir = view_dir;

                auto src = surf.sample(sr, refl_dir, pdf, samp);

                auto zero_pdf = pdf <= S(0.0);
#if 1
                auto emissive = dynamic_cast<emissive_type*>(surf.material.material_);
#else
                auto emissive = has_emissive_material(surf);
#endif

                src = mul( src, dot(n, refl_dir) / pdf, !emissive, src ); // TODO: maybe have emissive material return refl_dir so that dot(N,R) = 1?
                dst = mul( dst, src, !zero_pdf, dst );
                dst = select( zero_pdf, C(0.0), dst );

                if (emissive)
                {
                    *it = ~(*it);
                    return;
                }

                auto isect_pos = ray.ori + ray.dir * hit_rec.t;

                ray.ori = isect_pos + refl_dir * S(params.epsilon);
                ray.dir = refl_dir;
            }
#if defined(__CUDACC__) &&defined(SORT_BY_MATERIAL_TYPE)
          , stream
#endif
            );
    }

    template <typename InputIt, typename OutputIt>
    inline OutputIt compact(
            InputIt     first,
            InputIt     last,
            OutputIt    out
            )
    {
#ifdef __CUDACC__
        auto last_active = thrust::copy_if(
            thrust::device,
            first,
            last,
            out,
            [] __device__ (int index) { return index >= 0; }
            );


        thrust::copy_if(
            thrust::device,
            first,
            last,
            last_active,
            [] __device__ (int index) { return index < 0; }
            );

        return last_active;
#else
        ptrdiff_t l = 0;
        ptrdiff_t r = last - first - 1;

        for (auto it = first; it != last; ++it)
        {
            if (*it >= 0)
            {
                out[l++] = *it;
            }
            else
            {
                out[r--] = *it;
            }
        }

        assert(l == r + 1);

        return out + l;
#endif
    }

    void swap_index_buffers()
    {
#ifdef __CUDACC__
        thrust::copy(
                thrust::device,
                indices_back.begin(),
                indices_back.end(),
                indices.begin()
                );
#else
        std::copy(
                indices_back.begin(),
                indices_back.end(),
                indices.begin()
                );
#endif
    }

    template <typename IndexIt, typename Is>
    inline void terminate_active(
            IndexIt     indices_first,
            IndexIt     indices_last,
            Is          throughputs
            )
    {
        parallel_for(0, indices_last - indices_first,
            [=] __device__ (int index)
            {
                auto it = indices_first + index;

                if (*it >= 0)
                {
                    throughputs[*it] = typename std::iterator_traits<Is>::value_type(0.0f);
                }
            }
            );
    }

    template <typename IndexIt, typename Is, typename RT>
    inline void blend(
            IndexIt     indices_first,
            IndexIt     indices_last,
            Is          throughputs,
            RT          rt,
            unsigned    frame_num
            )
    {
        float alpha = 1.0f / frame_num;
        float sfactor = alpha;
        float dfactor = 1.0f - alpha;

        auto colors = rt.color();

        parallel_for(0, indices_last - indices_first,
            [=] __device__ (int index)
            {
                auto it = indices_first + index;

                if (*it < 0)
                {
                    *it = ~(*it);
                }

                auto& dst = colors[*it];
                auto color = to_rgba(throughputs[*it]);
     
                dst = color * sfactor + dst * dfactor;
            }
            );
    }
};

} // visioinaray

#endif // VSNRAY_EX_WAVEFRONT_PATHTRACER_PATHTRACER_H
