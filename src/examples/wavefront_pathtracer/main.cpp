// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>

#include <boost/filesystem.hpp>

#include <GL/glew.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/algorithm.h>
#include <visionaray/detail/parallel_algorithm.h>
#include <visionaray/detail/platform.h>

#include <visionaray/bvh.h>
#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/generic_material.h>
#include <visionaray/kernels.h>
#include <visionaray/material.h>
#include <visionaray/random_sampler.h>
#include <visionaray/scheduler.h>
#include <visionaray/spectrum.h>

#ifdef __CUDACC__
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>

#include <common/make_materials.h>
#include <common/model.h>
#include <common/obj_loader.h>
#include <common/pnm_image.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

#include "pathtracer.h"

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using material0                 = emissive<float>;
    using material1                 = matte<float>;
    using material2                 = mirror<float>;
    using material3                 = plastic<float>;

    using primitive_type            = model::triangle_type;
    using normal_type               = model::normal_type;
    using tex_coord_type            = model::tex_coord_type;
    using material_type             = generic_material<material0, material1, material2, material3>;

    using host_render_target_type   = cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using host_bvh_type             = index_bvh<primitive_type>;

#ifdef __CUDACC__
    using device_render_target_type = gpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
//    using device_render_target_type = pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using device_bvh_type           = cuda_index_bvh<primitive_type>;
    using device_tex_type           = cuda_texture<vector<4, unorm<8>>, 2>;
    using device_tex_ref_type       = typename device_tex_type::ref_type;
#endif
    renderer()
        : viewer_type(512, 512, "Visionaray Wavefront Pathtracer Example")
    {
        using namespace support;

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "filename",
            cl::Desc("Input file in wavefront obj format"),
            cl::Positional,
            cl::Required,
            cl::init(this->filename)
            ) );

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "camera",
            cl::Desc("Text file with camera parameters"),
            cl::ArgRequired,
            cl::init(this->initial_camera)
            ) );

        add_cmdline_option( cl::makeOption<bvh_build_strategy&>({
                { "default",            Binned,         "Binned SAH" },
                { "split",              Split,          "Binned SAH with spatial splits" }
            },
            "bvh",
            cl::Desc("BVH build strategy"),
            cl::ArgRequired,
            cl::init(this->builder)
            ) );
    }

    enum bvh_build_strategy
    {
        Binned = 0,  // Binned SAH builder, no spatial splits
        Split        // Split BVH, also binned and with SAH
    };

    camera                                      cam;
#ifdef __CUDACC__
    device_render_target_type                   rendertarget;
#else
    host_render_target_type                     rendertarget;
#endif
    bvh_build_strategy                          builder         = Binned;

    std::string                                 filename;
    std::string                                 initial_camera;

    pathtracer                                  tracer;

    model mod;
    index_bvh<model::triangle_type>             host_bvh;
    aligned_vector<material_type>               host_materials;

#ifdef SORT_BY_MATERIAL_TYPE
    aligned_vector<material0>                   host_material0s;
    aligned_vector<material1>                   host_material1s;
    aligned_vector<material2>                   host_material2s;
    aligned_vector<material3>                   host_material3s;
#endif

#ifdef __CUDACC__
    cuda_index_bvh<model::triangle_type>        device_bvh;
    thrust::device_vector<normal_type>          device_normals;
    thrust::device_vector<tex_coord_type>       device_tex_coords;
    thrust::device_vector<material_type>        device_materials;
    std::map<std::string, device_tex_type>      device_texture_map;
    thrust::device_vector<device_tex_ref_type>  device_textures;

#ifdef SORT_BY_MATERIAL_TYPE
    thrust::device_vector<material0>            device_material0s;
    thrust::device_vector<material1>            device_material1s;
    thrust::device_vector<material2>            device_material2s;
    thrust::device_vector<material3>            device_material3s;
#endif

#endif

    unsigned                                    frame_num       = 0;

    frame_counter                               counter;

protected:

    void on_display();
    void on_key_press(visionaray::key_event const& event);
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);

private:

    void clear_frame();

};


//-------------------------------------------------------------------------------------------------
// Struct with parameters
//

template <typename Primitives, typename Materials, typename Textures>
struct pathtracer_parameters
{
    using has_normals       = void;
    using has_textures      = void;

    using normal_binding    = normals_per_face_binding;
    using primitive_type    = typename std::iterator_traits<Primitives>::value_type;
    using normal_type       = renderer::normal_type;
    using tex_coords_type   = renderer::tex_coord_type;
    using material_type     = typename std::iterator_traits<Materials>::value_type;
    using texture_type      = typename std::iterator_traits<Textures>::value_type;
    using color_type        = vector<3, float>;

    using color_binding     = unspecified_binding;
    using light_type        = struct {};

    struct
    {
        Primitives begin;
        Primitives end;
    } prims;

    normal_type const* normals;
    tex_coords_type const* tex_coords;
    Materials materials;
    Textures textures;

    unsigned num_bounces;
    float epsilon;

    vec4 bg_color;
    vec4 ambient_color;
};

template <typename Primitives, typename Materials, typename Textures>
auto make_parameters(
        Primitives const&               begin,
        Primitives const&               end,
        renderer::normal_type const*    normals,
        renderer::tex_coord_type const* tex_coords,
        Materials                       materials,
        Textures const&                 textures,
        unsigned                        num_bounces,
        float                           epsilon,
        vec4 const&                     bg_color,
        vec4 const&                     ambient_color
        )
    -> pathtracer_parameters<Primitives, Materials, Textures>
{
    return pathtracer_parameters<Primitives, Materials, Textures>{
            { begin, end },
            normals,
            tex_coords,
            materials,
            textures,
            num_bounces,
            epsilon,
            bg_color,
            ambient_color
            };
}

template <typename Primitives, typename Textures>
auto make_parameters(
        Primitives const&               begin,
        Primitives const&               end,
        renderer::normal_type const*    normals,
        renderer::tex_coord_type const* tex_coords,
        renderer::material0 const*      mat0s,
        renderer::material1 const*      mat1s,
        renderer::material2 const*      mat2s,
        renderer::material3 const*      mat3s,
        Textures const&                 textures,
        unsigned                        num_bounces,
        float                           epsilon,
        vec4 const&                     bg_color,
        vec4 const&                     ambient_color
        )
    -> std::tuple<
        pathtracer_parameters<Primitives, renderer::material0 const*, Textures>,
        pathtracer_parameters<Primitives, renderer::material1 const*, Textures>,
        pathtracer_parameters<Primitives, renderer::material2 const*, Textures>,
        pathtracer_parameters<Primitives, renderer::material3 const*, Textures>
        >
{
#define ENTRY__(TYPE, OBJ)                                                              \
            pathtracer_parameters<Primitives, TYPE const*, Textures>{                   \
                { begin, end },                                                         \
                normals,                                                                \
                tex_coords,                                                             \
                OBJ,                                                                    \
                textures,                                                               \
                num_bounces,                                                            \
                epsilon,                                                                \
                bg_color,                                                               \
                ambient_color                                                           \
                }

    return std::make_tuple(
            ENTRY__(renderer::material0, mat0s),
            ENTRY__(renderer::material1, mat1s),
            ENTRY__(renderer::material2, mat2s),
            ENTRY__(renderer::material3, mat3s)
            );

#undef ENTRY__
}


//-------------------------------------------------------------------------------------------------
// I/O utility for camera lookat only - not fit for the general case!
//

std::istream& operator>>(std::istream& in, camera& cam)
{
    vec3 eye;
    vec3 center;
    vec3 up;

    in >> eye >> std::ws >> center >> std::ws >> up >> std::ws;
    cam.look_at(eye, center, up);

    return in;
}

std::ostream& operator<<(std::ostream& out, camera const& cam)
{
    out << cam.eye() << '\n';
    out << cam.center() << '\n';
    out << cam.up() << '\n';
    return out;
}


//-------------------------------------------------------------------------------------------------
// Clear frame buffer and reset frame counter
//

void renderer::clear_frame()
{
    frame_num = 0;

    rendertarget.clear_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // some setup

    using bvh_ref = index_bvh<model::triangle_type>::bvh_ref;

    std::vector<bvh_ref> bvhs;
    bvhs.push_back(host_bvh.ref());

    auto prims_begin = bvhs.data();
    auto prims_end   = bvhs.data() + bvhs.size();

    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = 10U;
    auto epsilon    = std::max( 1E-3f, length(diagonal) * 1E-5f );
//    auto amb        = ambient.x >= 0.0f // if set via cmdline
//                            ? vec4(ambient, 1.0f)
//                            : vec4(1.0)
//                            ;
    auto amb        = vec4(1.0);

#ifdef __CUDACC__
    thrust::device_vector<renderer::device_bvh_type::bvh_ref> device_primitives;

    device_primitives.push_back(device_bvh.ref());

    auto params = make_parameters(
            thrust::raw_pointer_cast(device_primitives.data()),
            thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
            thrust::raw_pointer_cast(device_normals.data()),
            thrust::raw_pointer_cast(device_tex_coords.data()),
#if defined(SORT_BY_MATERIAL_TYPE)
            thrust::raw_pointer_cast(device_material0s.data()),
            thrust::raw_pointer_cast(device_material1s.data()),
            thrust::raw_pointer_cast(device_material2s.data()),
            thrust::raw_pointer_cast(device_material3s.data()),
#else
            thrust::raw_pointer_cast(device_materials.data()),
#endif
            thrust::raw_pointer_cast(device_textures.data()),
            bounces,
            epsilon,
            vec4(background_color(), 1.0f),
            amb
            );

#else
    auto params = make_parameters(
            prims_begin,
            prims_end,
            mod.geometric_normals.data(),
            mod.tex_coords.data(),
#if defined(SORT_BY_MATERIAL_TYPE)
            host_material0s.data(),
            host_material1s.data(),
            host_material2s.data(),
            host_material3s.data(),
#else
            host_materials.data(),
#endif
            mod.textures.data(),
            bounces,
            epsilon,
            vec4(background_color(), 1.0f),
            amb
            );

#endif

    tracer.frame(params, rendertarget, cam, frame_num);

    // display the rendered image

    auto bgcolor = background_color();

    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_FRAMEBUFFER_SRGB);

    rendertarget.display_color_buffer();


    std::cout << "FPS: " << counter.register_frame() << '\r' << std::flush;

    if (frame_num == 10)
    {
        std::vector<vec4> tmp(rendertarget.width() * rendertarget.height());
#ifdef __CUDACC__
        cudaMemcpy(tmp.data(), (vec4*)rendertarget.color(),
                rendertarget.width() * rendertarget.height() * sizeof(vec4),
                cudaMemcpyDeviceToHost);
#endif
        std::vector<uint8_t> colors(tmp.size() * 3);
        for (int i = 0; i < tmp.size(); ++i)
        {
            colors[i * 3]     = (int)(tmp[i].x * 255.f);
            colors[i * 3 + 1] = (int)(tmp[i].y * 255.f);
            colors[i * 3 + 2] = (int)(tmp[i].z * 255.f);
        }
        pnm_image out_img(
            rendertarget.width(),
            rendertarget.height(),
            PF_RGB8,
            colors.data()
            );
        pnm_image::save_option opt1({"binary", true});
        boost::filesystem::path p(filename.c_str());
        out_img.save("/local/visionaray/build/" + p.filename().string() + ".pnm", {opt1});
    }
}


//-------------------------------------------------------------------------------------------------
// mouse handling
//

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.get_buttons() != mouse::NoButton)
    {
        clear_frame();
    }

    viewer_type::on_mouse_move(event);
}


//-------------------------------------------------------------------------------------------------
// keyboard handling
//

void renderer::on_key_press(key_event const& event)
{
    static const std::string camera_filename = "visionaray-camera.txt";

    switch (event.key())
    {
    case 'u':
        {
            std::ofstream file( camera_filename );
            if (file.good())
            {
                std::cout << "Storing camera to file: " << camera_filename << '\n';
                file << cam;
            }
        }
        break;

    case 'v':
        {
            std::ifstream file( camera_filename );
            if (file.good())
            {
                file >> cam;
                clear_frame();
                std::cout << "Load camera from file: " << camera_filename << '\n';
            }
        }
        break;

    default:
        break;
    }

    viewer_type::on_key_press(event);
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    tracer.resize(w, h);

    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rendertarget.resize(w, h);
    clear_frame();

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    renderer rend;

    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    try
    {
        visionaray::load_obj(rend.filename, rend.mod);
    }
    catch (std::exception& e)
    {
        std::cerr << "Failed loading obj model: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#if defined(SORT_BY_MATERIAL_TYPE)
    std::vector<size_t> material0_ids;
    std::vector<size_t> material1_ids;
    std::vector<size_t> material2_ids;
    std::vector<size_t> material3_ids;
    for (size_t i = 0; i < rend.mod.materials.size(); ++i)
    {
        auto mat = rend.mod.materials[i];std::cout << mat.name << '\n';
        // Add emissive material if emissive component > 0
        if (length(mat.ce) > 0.0f)
        {
            emissive<float> em;
            em.ce() = from_rgb(mat.ce);
            em.ls() = 1.0f;
            rend.host_material0s.emplace_back(em);
            material0_ids.push_back(i);
        }
        else if (mat.specular_exp < 0.001f)
        {
            matte<float> ma;
            ma.ca() = from_rgb(mat.ca);
            ma.cd() = from_rgb(mat.cd);
            ma.ka() = 1.0f;
            ma.kd() = 1.0f;
            rend.host_material1s.emplace_back(ma);
            material1_ids.push_back(i);
        }
        else if (mat.specular_exp > 100.0f)
        {
            mirror<float> mi;
            mi.cr() = from_rgb(mat.cs);
            mi.kr() = 1.0f;
            mi.ior() = from_rgb(mat.ior);
            mi.absorption() = from_rgb(mat.absorption);
            rend.host_material2s.emplace_back(mi);
            material2_ids.push_back(i);
        }
        else
        {
            plastic<float> pl;
            pl.ca() = from_rgb(mat.ca);
            pl.cd() = from_rgb(mat.cd);
            pl.cs() = from_rgb(mat.cs);
            pl.ka() = 1.0f;
            pl.kd() = 1.0f;
            pl.ks() = 1.0f;
            pl.specular_exp() = mat.specular_exp;
            rend.host_material3s.emplace_back(pl);
            material3_ids.push_back(i);
        }
    }

    for (auto& p : rend.mod.primitives)
    {
        unsigned bits = 0x80000000;

        auto it = std::find(material0_ids.begin(), material0_ids.end(), p.geom_id);
        if (it != material0_ids.end())
        {
            auto new_geom_id = it - material0_ids.begin();
            new_geom_id |= bits;
            p.geom_id = new_geom_id;
            continue;
        }
        bits >>= 1;

        it = std::find(material1_ids.begin(), material1_ids.end(), p.geom_id);
        if (it != material1_ids.end())
        {
            auto new_geom_id = it - material1_ids.begin();
            new_geom_id |= bits;
            p.geom_id = new_geom_id;
            continue;
        }
        bits >>= 1;

        it = std::find(material2_ids.begin(), material2_ids.end(), p.geom_id);
        if (it != material2_ids.end())
        {
            auto new_geom_id = it - material2_ids.begin();
            new_geom_id |= bits;
            p.geom_id = new_geom_id;
            continue;
        }
        bits >>= 1;

        it = std::find(material3_ids.begin(), material3_ids.end(), p.geom_id);
        if (it != material3_ids.end())
        {
            auto new_geom_id = it - material3_ids.begin();
            new_geom_id |= bits;
            p.geom_id = new_geom_id;
            continue;
        }
        bits >>= 1;
    }
#else
    // Convert generic materials to viewer's material type
    rend.host_materials = make_materials(
            renderer::material_type{},
            rend.mod.materials,
            [](aligned_vector<renderer::material_type>& cont, model::material_type mat)
            {
                // Add emissive material if emissive component > 0
                if (length(mat.ce) > 0.0f)
                {
                    emissive<float> em;
                    em.ce() = from_rgb(mat.ce);
                    em.ls() = 1.0f;
                    cont.emplace_back(em);
                }
                else if (mat.specular_exp < 0.001f)
                {
                    matte<float> ma;
                    ma.ca() = from_rgb(mat.ca);
                    ma.cd() = from_rgb(mat.cd);
                    ma.ka() = 1.0f;
                    ma.kd() = 1.0f;
                    cont.push_back(ma);
                }
                else if (mat.specular_exp > 100.0f)
                {
                    mirror<float> mi;
                    mi.cr() = from_rgb(mat.cs);
                    mi.kr() = 1.0f;
                    mi.ior() = from_rgb(mat.ior);
                    mi.absorption() = from_rgb(mat.absorption);
                    cont.push_back(mi);
                }
                else
                {
                    plastic<float> pl;
                    pl.ca() = from_rgb(mat.ca);
                    pl.cd() = from_rgb(mat.cd);
                    pl.cs() = from_rgb(mat.cs);
                    pl.ka() = 1.0f;
                    pl.kd() = 1.0f;
                    pl.ks() = 1.0f;
                    pl.specular_exp() = mat.specular_exp;
                    cont.emplace_back(pl);
                }
            }
            );
#endif

    std::cout << "Creating BVH...\n";

    // Create the BVH on the host
    rend.host_bvh = build<index_bvh<model::triangle_type>>(
            rend.mod.primitives.data(),
            rend.mod.primitives.size(),
            rend.builder == renderer::Split
            );

    std::cout << "Ready\n";

#ifdef __CUDACC__

    // Copy data to GPU
    TRY_ALLOC(rend.device_bvh = renderer::device_bvh_type(rend.host_bvh));
    TRY_ALLOC(rend.device_normals = rend.mod.geometric_normals);
    TRY_ALLOC(rend.device_tex_coords = rend.mod.tex_coords);
    TRY_ALLOC(rend.device_materials = rend.host_materials);
#if defined(SORT_BY_MATERIAL_TYPE)
    TRY_ALLOC(rend.device_material0s = rend.host_material0s);
    TRY_ALLOC(rend.device_material1s = rend.host_material1s);
    TRY_ALLOC(rend.device_material2s = rend.host_material2s);
    TRY_ALLOC(rend.device_material3s = rend.host_material3s);
#endif

    // Copy textures and texture references to the GPU

    TRY_ALLOC(rend.device_textures.resize(rend.mod.textures.size()));

    for (auto const& pair_host_tex : rend.mod.texture_map)
    {
        auto const& host_tex = pair_host_tex.second;
        renderer::device_tex_type device_tex(pair_host_tex.second);
        auto const& p = rend.device_texture_map.emplace(pair_host_tex.first, std::move(device_tex));

        assert(p.second /* inserted */);

        auto it = p.first;

        // Texture references ensure that we don't allocate storage
        // for the same texture map more than once.
        // By checking if the pointer in the ref contains the
        // address of the first texel of the map, we can identify
        // which texture_ref references which texture and recreate
        // that relation on the GPU.
        for (size_t i = 0; i < rend.mod.textures.size(); ++i)
        {
            if (rend.mod.textures[i].data() == host_tex.data())
            {
                rend.device_textures[i] = renderer::device_tex_ref_type(it->second);
            }
        }
    }

    // Place some dummy textures where geometry has no texture
    for (size_t i = 0; i < rend.mod.textures.size(); ++i)
    {
        if (rend.mod.textures[i].width() == 0 || rend.mod.textures[i].height() == 0)
        {
            vector<4, unorm<8>>* dummy = nullptr;
            renderer::device_tex_type device_tex(dummy, 0, 0, Clamp, Nearest);

            // Try to insert the dummy texture into the
            // device texture map...
            auto p = rend.device_texture_map.emplace("", std::move(device_tex));

            // ... but maybe a dummy texture was already
            // inserted, then just find that
            if (!p.second)
            {
                auto it = rend.device_texture_map.find("");
                rend.device_textures[i] = renderer::device_tex_ref_type(it->second);

            }
            else
            {
                auto it = p.first;
                rend.device_textures[i] = renderer::device_tex_ref_type(it->second);
            }
        }
    }
#endif

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);

    // Load camera from file or set view-all
    std::ifstream file(rend.initial_camera);
    if (file.good())
    {
        file >> rend.cam;
    }
    else
    {
        rend.cam.view_all( rend.mod.bbox );
    }

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
