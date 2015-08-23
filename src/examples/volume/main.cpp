// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <memory>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

using std::make_shared;
using std::shared_ptr;

using namespace visionaray;

using manipulators  = std::vector<shared_ptr<visionaray::camera_manipulator>>;
using viewer_type   = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Texture data
//

// volume data
static const float voldata[2 * 2 * 2] = {

        // slice 1
        1.0f, 0.0f,
        0.0f, 1.0f,

        // slice 2
        0.0f, 1.0f,
        1.0f, 0.0f

        };

// post-classification transfer function
static const vec4 tfdata[4 * 4] = {
        { 0.0f, 0.0f, 0.0f, 0.02f },
        { 0.7f, 0.1f, 0.2f, 0.03f },
        { 0.1f, 0.9f, 0.3f, 0.04f },
        { 1.0f, 1.0f, 1.0f, 0.05f }
        };


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<simd::float4>;

    renderer()
        : viewer_type(512, 512, "Visionaray Volume Rendering Example")
        , bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , host_sched(8)
        , down_button(mouse::NoButton)
        , volume({2, 2, 2})
        , transfunc({4})
    {
        volume.set_data(voldata);
        volume.set_filter_mode(Nearest);
        volume.set_address_mode(Clamp);

        transfunc.set_data(tfdata);
        transfunc.set_filter_mode(Linear);
        transfunc.set_address_mode(Clamp);
    }

    aabb                                        bbox;
    camera                                      cam;
    manipulators                                manips;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    mouse::button down_button;
    mouse::pos motion_pos;
    mouse::pos down_pos;
    mouse::pos up_pos;


    // texture references

    texture_ref<float, NormalizedFloat, 3> volume;
    texture_ref<vec4, ElementType, 1> transfunc;

protected:

    void on_display();
    void on_resize(int w, int h);

};

std::unique_ptr<renderer> rend(nullptr);


//-------------------------------------------------------------------------------------------------
// Display function, implements the volume rendering algorithm
//

void renderer::on_display()
{
    // some setup

    using R = renderer::host_ray_type;
    using S = R::scalar_type;
    using C = vector<4, S>;

    auto sparams = make_sched_params<pixel_sampler::uniform_type>(
            rend->cam,
            rend->host_rt
            );


    // call kernel in schedulers' frame() method

    rend->host_sched.frame([&](R ray) -> result_record<S>
    {
        result_record<S> result;

        auto hit_rec = intersect(ray, bbox);
        auto t = hit_rec.tnear;

        result.color = C(0.0);

        while ( any(t < hit_rec.tfar) )
        {
            auto pos = ray.ori + ray.dir * t;
            auto tex_coord = vector<3, S>(
                    ( pos.x + 1.0f ) / 2.0f,
                    (-pos.y + 1.0f ) / 2.0f,
                    (-pos.z + 1.0f ) / 2.0f
                    );

            // sample volume and do post-classification
            auto voxel = tex3D(rend->volume, tex_coord);
            C color = tex1D(rend->transfunc, voxel);

            // premultiplied alpha
            auto premult = color.xyz() * color.w;
            color = C(premult, color.w);

            // front-to-back alpha compositing
            result.color += select(
                    t < hit_rec.tfar,
                    color * (1.0f - result.color.w),
                    C(0.0)
                    );

            // early-ray termination - don't traverse w/o a contribution
            if ( all(result.color.w >= 0.999) )
            {
                break;
            }

            // step on
            t += 0.01f;
        }

        result.hit = hit_rec.hit;
        return result;
    }, sparams);


    // display the rendered image

    auto bgcolor = rend->background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    rend->host_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    rend->cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    rend->cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend->host_rt.resize(w, h);

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    rend = std::unique_ptr<renderer>(new renderer);

    try
    {
        rend->init(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    float aspect = rend->width() / static_cast<float>(rend->height());

    rend->cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend->cam.view_all( rend->bbox );

    rend->add_manipulator( std::make_shared<arcball_manipulator>(rend->cam, mouse::Left) );
    rend->add_manipulator( std::make_shared<pan_manipulator>(rend->cam, mouse::Middle) );
    rend->add_manipulator( std::make_shared<zoom_manipulator>(rend->cam, mouse::Right) );

    rend->event_loop();
}
