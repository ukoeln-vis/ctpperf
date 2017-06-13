#pragma once

#include <visionaray/material.h>

#include "material_base.h"

namespace visionaray
{

struct emissive_type : material_base
{
    using SR = material_base::SR;

    emissive_type(emissive<float> const& mat)
        : mat_(mat)
    {
    }

    spectrum<float> ambient() const;
    spectrum<float> shade(SR const& sr) const;
    spectrum<float> sample(
            SR const& sr,
            vec3& refl_dir,
            float& pdf,
            random_sampler<float>& samp
            ) const;

    emissive<float> mat_;

};

} // visionaray
