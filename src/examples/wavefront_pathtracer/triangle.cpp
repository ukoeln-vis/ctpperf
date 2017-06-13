#include <visionaray/bvh.h>
#include <visionaray/get_tex_coord.h>

#include "triangle.h"

namespace visionaray
{

aabb triangle_prim::get_bounds_impl()
{
    return get_bounds(tri_);
}

hit_record<ray, primitive<unsigned>> triangle_prim::intersect_impl(ray r)
{
    return visionaray::intersect(r, tri_);
}

vec3 triangle_prim::get_normal_impl(hit_record<ray, primitive<unsigned>> const& hr)
{
    return normalize(cross(tri_.e1, tri_.e2));
}

vec2 triangle_prim::get_tex_coord_impl(vec2 const* tex_coords, hit_record<ray, primitive<unsigned>> const& hr)
{
    //return get_tex_coord(tex_coords, hr, basic_triangle<3, float>{});
    return vec2(0.f);
}

void triangle_prim::split_primitive_impl(aabb& L, aabb& R, float plane, int axis)
{
    split_primitive(L, R, plane, axis, tri_);
}

} // visionaray
