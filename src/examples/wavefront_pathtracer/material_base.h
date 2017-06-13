
#pragma once

#include <visionaray/math/math.h>
#include <visionaray/point_light.h>
#include <visionaray/random_sampler.h>
#include <visionaray/shade_record.h>
#include <visionaray/spectrum.h>

namespace visionaray
{

struct material_base
{
    using L = point_light<float>;
    using SR = shade_record<L, vec3, float>;
    virtual spectrum<float> ambient() const = 0;
    virtual spectrum<float> shade(SR const& sr) const = 0;
    virtual spectrum<float> sample(
            SR const& sr,
            vec3& refl_dir,
            float& pdf,
            random_sampler<float>& samp
            ) const = 0;
};

// Visionaray internal functions call
//  mat.shade(), mat.sample(), etc.
// We can however not store references in a std::vector<>,
// and pointers would require mat->shade() syntax.
// Thus use a wrapper that stores a pointer internally,
// with an interface that is compabile with Visionaray.
// We try to ensure zero overhead by forcing the interface
// functions inline!
struct material_wrapper
{
    using scalar_type = float;
    using SR = material_base::SR;

    material_wrapper() = default;
    material_wrapper(material_base* material)
        : material_(material)
    {
    }

    VSNRAY_FORCE_INLINE spectrum<float> shade(SR const& sr) const
    {
        return material_->shade(sr);
    }

    VSNRAY_FORCE_INLINE spectrum<float> sample(
            SR const& sr,
            vec3& refl_dir,
            float& pdf,
            random_sampler<float>& samp
            ) const
    {
        return material_->sample(sr, refl_dir, pdf, samp);
    }

    material_base* material_;
};

} // visionaray
