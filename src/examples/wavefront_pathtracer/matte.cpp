
#include "matte.h"

namespace visionaray
{

spectrum<float> matte_type::ambient() const
{
    return mat_.ambient();
}

spectrum<float> matte_type::shade(matte_type::SR const& sr) const
{
    return mat_.shade(sr);
}

spectrum<float> matte_type::sample(
        matte_type::SR const& sr,
        vec3& refl_dir,
        float& pdf,
        random_sampler<float>& samp
        ) const
{
    return mat_.sample(sr, refl_dir, pdf, samp);
}

}
