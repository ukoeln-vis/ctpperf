
#include "emissive.h"

namespace visionaray
{

spectrum<float> emissive_type::ambient() const
{
    return mat_.ambient();
}

spectrum<float> emissive_type::shade(emissive_type::SR const& sr) const
{
    return mat_.shade(sr);
}

spectrum<float> emissive_type::sample(
        emissive_type::SR const& sr,
        vec3& refl_dir,
        float& pdf,
        random_sampler<float>& samp
        ) const
{
    return mat_.sample(sr, refl_dir, pdf, samp);
}

}
