#include <visionaray/bvh.h>

#include "sphere.h"

namespace visionaray
{

aabb sphere_prim::get_bounds_impl()
{
    return get_bounds(sph_);
}

hit_record<ray, primitive<unsigned>> sphere_prim::intersect_impl(ray r)
{
    return visionaray::intersect(r, sph_);
}

vec3 sphere_prim::get_normal_impl(hit_record<ray, primitive<unsigned>> const& hr)
{
    return (hr.isect_pos - sph_.center) / sph_.radius;
}

vec2 sphere_prim::get_tex_coord_impl(vec2 const* tex_coords, hit_record<ray, primitive<unsigned>> const& hr)
{
    return vec2(0.0f);
}

void sphere_prim::split_primitive_impl(aabb& L, aabb& R, float plane, int axis)
{
    split_primitive(L, R, plane, axis, sph_);
}

} // visionaray
