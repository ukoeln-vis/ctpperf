#pragma once

#include "primitive_base.h"

namespace visionaray
{

struct sphere_prim : primitive_base
{
    sphere_prim(basic_sphere<float> const& sph)
        : sph_(sph)
    {
    }

    aabb get_bounds_impl();
    hit_record<ray, primitive<unsigned>> intersect_impl(ray r);
    vec3 get_normal_impl(hit_record<ray, primitive<unsigned>> const& hr);
    vec2 get_tex_coord_impl(vec2 const* tex_coords, hit_record<ray, primitive<unsigned>> const& hr);
    void split_primitive_impl(aabb& L, aabb& R, float plane, int axis);

    basic_sphere<float> sph_;
};

} // visionaray
