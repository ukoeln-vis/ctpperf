#pragma once

#include <visionaray/material.h>

#include "material_base.h"

namespace visionaray
{

struct plastic_type : material_base
{
    using SR = material_base::SR;

    plastic_type(plastic<float> const& mat)
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

    plastic<float> mat_;

};

} // visionaray
