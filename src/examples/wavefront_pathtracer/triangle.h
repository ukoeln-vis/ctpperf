#pragma once

#include "primitive_base.h"

namespace visionaray
{

struct triangle_prim : primitive_base
{
    triangle_prim(basic_triangle<3, float> const& tri)
        : tri_(tri)
    {
    }

    aabb get_bounds_impl();
    hit_record<ray, primitive<unsigned>> intersect_impl(ray r);
    vec3 get_normal_impl(hit_record<ray, primitive<unsigned>> const& hr);
    vec2 get_tex_coord_impl(vec2 const* tex_coords, hit_record<ray, primitive<unsigned>> const& hr);
    void split_primitive_impl(aabb& L, aabb& R, float plane, int axis);

    basic_triangle<3, float> tri_;
};

} // visionaray
