
#pragma once

#include <visionaray/math/math.h>
#include <visionaray/prim_traits.h>

namespace visionaray
{

struct primitive_base
{
    virtual aabb get_bounds_impl() = 0;
    virtual hit_record<ray, primitive<unsigned>> intersect_impl(ray r) = 0;
    virtual vec3 get_normal_impl(hit_record<ray, primitive<unsigned>> const& hr) = 0;
    virtual vec2 get_tex_coord_impl(vec2 const* tex_coords, hit_record<ray, primitive<unsigned>> const& hr) = 0;
    virtual void split_primitive_impl(aabb& L, aabb& R, float plane, int axis) = 0;
};

inline hit_record<ray, primitive<unsigned>> intersect(ray r, primitive_base* prim)
{
    return prim->intersect_impl(r);
}

inline aabb get_bounds(primitive_base* prim)
{
    return prim->get_bounds_impl();
}

inline vec3 get_normal(
        hit_record<ray, primitive<unsigned>> const& hr,
        primitive_base*             prim
        )
{
    return prim->get_normal_impl(hr);
}

inline vec3 get_shading_normal(
        hit_record<ray, primitive<unsigned>> const& hr,
        primitive_base*             prim
        )
{
    return prim->get_normal_impl(hr);
}

inline vec2 get_tex_coord(
        vec2 const*                                   tex_coords,
        hit_record<ray, primitive<unsigned>> const& hr,
        primitive_base* prim
        )
{
    return vec2(0.0f);
    //return prim->get_tex_coord_impl(tex_coords, hr);
}

inline void split_primitive(aabb& L, aabb& R, float plane, int axis, primitive_base* prim)
{
    return prim->split_primitive_impl(L, R, plane, axis);
}

template <typename Binding>
struct num_normals<primitive_base*, Binding>
{
    enum { value = 0 }; // TODO!
};

} // visionaray
