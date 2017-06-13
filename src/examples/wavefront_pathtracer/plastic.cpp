
#include "plastic.h"

namespace visionaray
{

spectrum<float> plastic_type::ambient() const
{
    return mat_.ambient();
}

spectrum<float> plastic_type::shade(plastic_type::SR const& sr) const
{
    return mat_.shade(sr);
}

spectrum<float> plastic_type::sample(
        plastic_type::SR const& sr,
        vec3& refl_dir,
        float& pdf,
        random_sampler<float>& samp
        ) const
{
    return mat_.sample(sr, refl_dir, pdf, samp);
}

}
