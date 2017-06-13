
#include "mirror.h"

namespace visionaray
{

spectrum<float> mirror_type::ambient() const
{
    return mat_.ambient();
}

spectrum<float> mirror_type::shade(mirror_type::SR const& sr) const
{
    return mat_.shade(sr);
}

spectrum<float> mirror_type::sample(    
        mirror_type::SR const& sr,
        vec3& refl_dir,
        float& pdf,
        random_sampler<float>& samp
        ) const
{
    return mat_.sample(sr, refl_dir, pdf, samp);
}

}
