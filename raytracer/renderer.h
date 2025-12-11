#ifndef RT_RENDERER_H
#define RT_RENDERER_H
//==============================================================================================
// Renderer - Qt integration wrapper for RTiOW ray tracer
// Bridges graphics::Camera (GL) to RTiOW scene rendering with QImage output
// Supports both real-time preview and batch animation rendering
//==============================================================================================

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"

#include "../graphics/Camera.h"
#include "../graphics/Light.h"
#include "../math/Mathematics.h"

#include <QImage>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <functional>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

namespace rt {

// Render mode for batch rendering
enum class BatchRenderMode {
    CUDA_SOFTWARE,
    OPTIX_HARDWARE,
    CPU_RAYTRACER
};

// Render settings for animation
struct RenderSettings {
    int width = 800;
    int height = 600;
    int samples_per_pixel = 10;
    int max_depth = 10;
    bool use_bvh = true;
    int start_frame = 1;
    int end_frame = 30;
    QString output_directory = "./output";
    QString filename_pattern = "frame_%1.png";  // Qt-style placeholder for arg()
    BatchRenderMode render_mode = BatchRenderMode::CUDA_SOFTWARE;  // Which renderer to use

    // Animation settings
    bool animate_camera = true;      // Orbit camera around scene
    bool animate_light = true;       // Rotate light around scene
    double camera_orbit_radius = 5.0;  // Distance from origin
    double camera_height = 2.0;        // Camera Y position
    double light_orbit_radius = 3.0;   // Light orbit radius
    double light_height = 3.0;         // Light Y position

    // Time-based animation (independent of frame count)
    double framerate = 30.0;                    // Target framerate for time calculation
    double camera_rotation_speed = 0.4189;      // Radians per second (2π/15 = full rotation in 15 seconds)

    // Multithreading
    bool use_multithreading = true;  // Enable/disable multithreading
    int num_threads = 0;  // 0 = auto-detect (use all cores)

    // FFmpeg pipe mode (skip PNG files, pipe directly to FFmpeg)
    bool use_ffmpeg_pipe = false;  // Enable direct FFmpeg pipe mode
    bool save_frame_data = true;   // Save per-frame timing data to JSON file
};


// Progress callback: (current_frame, total_frames)
using ProgressCallback = std::function<void(int, int)>;

// Frame complete callback: (frame_time_seconds)
using FrameCompleteCallback = std::function<void(double)>;

// Completion callback: (success, message)
using CompletionCallback = std::function<void(bool, const QString&)>;


class Renderer {
  public:
    Renderer() : cancel_requested(false) {}

    // Set the scene (BVH root or hittable_list)
    void set_scene(shared_ptr<hittable> scene) {
        world = scene;
    }

    // Set lights for direct illumination
    void set_lights(const std::vector<graphics::Lightf>& scene_lights) {
        lights = scene_lights;
        std::clog << "[Renderer] set_lights called with " << lights.size() << " lights" << std::endl;
        for (size_t i = 0; i < lights.size(); i++) {
            std::clog << "  Light " << i << ": type=" << static_cast<int>(lights[i].type)
                      << " pos=(" << lights[i].position.x() << "," << lights[i].position.y() << "," << lights[i].position.z() << ")"
                      << " intensity=" << lights[i].intensity << std::endl;
        }
    }

    // Set render parameters
    void set_samples_per_pixel(int spp) { samples_per_pixel = spp; }
    void set_max_depth(int depth) { max_depth = depth; }
    int get_samples_per_pixel() const { return samples_per_pixel; }
    int get_max_depth() const { return max_depth; }

    // Cancel control
    void request_cancel() { cancel_requested = true; }
    void reset_cancel() { cancel_requested = false; }
    bool is_cancel_requested() const { return cancel_requested; }

    // Render single frame to QImage using graphics::Camera (for real-time preview)
    void render_frame(QImage& image, const graphics::Cameraf& gl_camera) {
        render_frame_internal(image, gl_camera);
    }

    // Batch render animation sequence (runs synchronously - call from worker thread)
    // Returns true if completed, false if cancelled
    bool render_animation(const RenderSettings& settings,
                          const graphics::Cameraf& base_camera,
                          ProgressCallback progress,
                          FrameCompleteCallback frame_complete = nullptr,
                          CompletionCallback completion = nullptr) {

        reset_cancel();

        // Create output directory if it doesn't exist
        QDir dir(settings.output_directory);
        if (!dir.exists()) {
            if (!dir.mkpath(".")) {
                if (completion) {
                    completion(false, QString("Failed to create output directory: %1").arg(settings.output_directory));
                }
                return false;
            }
        }

        // Determine number of threads
        int num_threads = 1;
        if (settings.use_multithreading) {
            num_threads = settings.num_threads;
            if (num_threads <= 0) {
                num_threads = std::thread::hardware_concurrency();
                if (num_threads == 0) num_threads = 4;  // Fallback
            }
        }

        // Store settings for this render
        int render_spp = settings.samples_per_pixel;
        int render_depth = settings.max_depth;

        QImage image(settings.width, settings.height, QImage::Format_RGB32);
        int total_frames = settings.end_frame - settings.start_frame + 1;

        std::clog << "[Renderer] Starting batch render: " << total_frames << " frames" << std::endl;
        std::clog << "[Renderer] Resolution: " << settings.width << "x" << settings.height << std::endl;
        std::clog << "[Renderer] Samples: " << render_spp << ", Depth: " << render_depth << std::endl;
        std::clog << "[Renderer] Threads: " << num_threads << std::endl;
        std::clog << "[Renderer] Animation: Camera=" << (settings.animate_camera ? "yes" : "no")
                  << ", Light=" << (settings.animate_light ? "yes" : "no") << std::endl;
        std::clog << "[Renderer] Output: " << settings.output_directory.toStdString() << std::endl;

        // Setup JSON frame data file
        QString animName = QDir(settings.output_directory).dirName();
        QString dataFilePath = settings.output_directory + "/" + animName + "_framedata.json";
        QFile dataFile(dataFilePath);
        QTextStream* dataStream = nullptr;
        if (dataFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            dataStream = new QTextStream(&dataFile);
            *dataStream << "[\n";  // Start JSON array
            std::clog << "[Renderer] Frame data will be saved to: " << dataFilePath.toStdString() << std::endl;
        }

        auto batch_start_time = std::chrono::high_resolution_clock::now();
        int framesRendered = 0;

        for (int frame = settings.start_frame; frame <= settings.end_frame; frame++) {
            // Check for cancellation
            if (cancel_requested) {
                std::clog << "[Renderer] Render cancelled at frame " << frame << std::endl;
                if (completion) {
                    completion(false, QString("Render cancelled at frame %1").arg(frame));
                }
                return false;
            }

            int frame_index = frame - settings.start_frame + 1;

            // Calculate time-based animation (independent of frame count)
            double delta_time = 1.0 / settings.framerate;  // Time per frame
            double elapsed_time = (frame - settings.start_frame) * delta_time;  // Total elapsed time
            double angle = elapsed_time * settings.camera_rotation_speed;  // Radians based on time

            // Create animated camera for this frame
            graphics::Cameraf frame_camera = base_camera;
            if (settings.animate_camera) {
                // Camera uses spherical coordinates (theta, phi, radius) around lookAt point
                // Start from the base camera's current rotation and animate theta (horizontal orbit)
                float base_theta = base_camera.getTheta();
                float base_phi = base_camera.getPhi();  // Keep the same vertical angle as GL camera
                float base_radius = base_camera.getRadius();

                float theta = base_theta + static_cast<float>(angle);  // Orbit from current position
                float phi = base_phi;  // Maintain the same vertical tilt as GL view

                frame_camera.setPosition(0.0f, 0.0f, 0.0f);  // Look at origin
                frame_camera.setRadius(base_radius);
                frame_camera.setRotation(theta, phi);
            }

            // Calculate animated light position
            point3 light_pos(3.0, 3.0, 3.0);  // Default
            if (settings.animate_light) {
                double light_x = settings.light_orbit_radius * std::cos(angle);
                double light_z = settings.light_orbit_radius * std::sin(angle);
                light_pos = point3(light_x, settings.light_height, light_z);
            }

            // Temporarily set render settings
            int saved_spp = samples_per_pixel;
            int saved_depth = max_depth;
            samples_per_pixel = render_spp;
            max_depth = render_depth;

            std::clog << "[Renderer] Starting frame " << frame_index << "/" << total_frames << "..." << std::endl;

            // Start timer RIGHT before render
            auto frame_start_time = std::chrono::high_resolution_clock::now();

            // Render frame with multithreading (no scanline progress - just frame level)
            render_frame_multithreaded(image, frame_camera, num_threads);

            // Stop timer RIGHT after render
            auto frame_end_time = std::chrono::high_resolution_clock::now();

            // Restore settings
            samples_per_pixel = saved_spp;
            max_depth = saved_depth;

            // Calculate frame render time
            auto frame_duration = std::chrono::duration<double>(frame_end_time - frame_start_time);
            double frame_seconds = frame_duration.count();

            std::clog << "[Renderer] Frame " << frame_index << "/" << total_frames << " rendered in "
                      << frame_seconds << " seconds (" << num_threads << " threads)" << std::endl;

            // Log frame data to JSON
            if (dataStream) {
                auto elapsed = std::chrono::duration<double>(frame_end_time - batch_start_time);
                double totalElapsed = elapsed.count();
                if (framesRendered > 0) *dataStream << ",\n";
                *dataStream << QString("  {\"frame\": %1, \"render_ms\": %2, \"total_elapsed_s\": %3, \"fps\": %4}")
                    .arg(frame)
                    .arg(frame_seconds * 1000.0, 0, 'f', 2)
                    .arg(totalElapsed, 0, 'f', 3)
                    .arg(1.0 / frame_seconds, 0, 'f', 1);
                framesRendered++;
            }

            // Update progress (frame level only)
            if (progress) {
                progress(frame_index, total_frames);
            }

            // Notify frame complete with timing
            if (frame_complete) {
                frame_complete(frame_seconds);
            }

            // Check for cancellation after render
            if (cancel_requested) {
                std::clog << "[Renderer] Render cancelled after frame " << frame << std::endl;
                if (completion) {
                    completion(false, QString("Render cancelled after frame %1").arg(frame));
                }
                return false;
            }

            // Save frame
            QString filename = QString(settings.filename_pattern).arg(frame, 4, 10, QChar('0'));
            QString filepath = settings.output_directory + "/" + filename;

            if (!image.save(filepath)) {
                std::cerr << "[Renderer] Failed to save: " << filepath.toStdString() << std::endl;
                if (completion) {
                    completion(false, QString("Failed to save frame: %1").arg(filepath));
                }
                return false;
            }

            std::clog << "[Renderer] Saved frame " << frame_index << "/" << total_frames
                      << ": " << filepath.toStdString() << std::endl;
        }

        auto batch_end_time = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::seconds>(batch_end_time - batch_start_time);

        // Close JSON file
        if (dataStream) {
            *dataStream << "\n]\n";  // End JSON array
            delete dataStream;
            dataFile.close();
            std::clog << "[Renderer] Frame data saved to: " << dataFilePath.toStdString() << std::endl;
        }

        std::clog << "[Renderer] Batch render complete! Total time: " << batch_duration.count() << " seconds" << std::endl;

        if (completion) {
            completion(true, QString("Render complete! %1 frames in %2 seconds")
                       .arg(total_frames).arg(batch_duration.count()));
        }

        return true;
    }

  private:
    shared_ptr<hittable> world;
    std::vector<graphics::Lightf> lights;
    int samples_per_pixel = 10;
    int max_depth = 10;
    std::atomic<bool> cancel_requested;

    // Camera state (computed from graphics::Camera)
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u, v, w;  // Camera basis vectors

    // Internal render function (single-threaded, for real-time preview - no progress needed)
    void render_frame_internal(QImage& image, const graphics::Cameraf& gl_camera) {
        int width = image.width();
        int height = image.height();

        // Initialize camera parameters from GL camera
        initialize_from_gl_camera(gl_camera, width, height);

        double pixel_samples_scale = 1.0 / samples_per_pixel;

        // Using scanLine() for faster direct memory access instead of setPixel()
        for (int j = 0; j < height; j++) {
            // Check for cancellation at each scanline
            if (cancel_requested) {
                return;
            }

            QRgb* scanLine = reinterpret_cast<QRgb*>(image.scanLine(j));
            for (int i = 0; i < width; i++) {
                color pixel_color(0, 0, 0);

                for (int s = 0; s < samples_per_pixel; s++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth);
                }

                // Average and gamma correct
                pixel_color = pixel_color * pixel_samples_scale;

                // Gamma correction (gamma = 2.0)
                double r_val = std::sqrt(pixel_color.x());
                double g_val = std::sqrt(pixel_color.y());
                double b_val = std::sqrt(pixel_color.z());

                // Clamp and convert to 8-bit
                int ir = int(256 * std::clamp(r_val, 0.0, 0.999));
                int ig = int(256 * std::clamp(g_val, 0.0, 0.999));
                int ib = int(256 * std::clamp(b_val, 0.0, 0.999));

                scanLine[i] = qRgb(ir, ig, ib);
            }
        }
    }

    // Multithreaded render function for batch rendering
    void render_frame_multithreaded(QImage& image, const graphics::Cameraf& gl_camera,
                                    int num_threads) {
        int width = image.width();
        int height = image.height();

        // Initialize camera parameters from GL camera (shared across threads)
        initialize_from_gl_camera(gl_camera, width, height);

        // Thread-local copies of render parameters
        const double pixel_samples_scale = 1.0 / samples_per_pixel;
        const int spp = samples_per_pixel;
        const int depth = max_depth;

        // Create thread workers
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        // Each thread renders a subset of scanlines
        // Using scanLine() for faster direct memory access instead of setPixel()
        auto render_scanlines = [&](int thread_id) {
            for (int j = thread_id; j < height; j += num_threads) {
                // Check for cancellation
                if (cancel_requested) {
                    return;
                }

                // Get scanline pointer once per row (thread-safe for different rows)
                QRgb* scanLine = reinterpret_cast<QRgb*>(image.scanLine(j));

                // Render this scanline
                for (int i = 0; i < width; i++) {
                    color pixel_color(0, 0, 0);

                    for (int s = 0; s < spp; s++) {
                        ray r = get_ray(i, j);
                        pixel_color += ray_color(r, depth);
                    }

                    // Average and gamma correct
                    pixel_color = pixel_color * pixel_samples_scale;

                    // Gamma correction (gamma = 2.0)
                    double r_val = std::sqrt(pixel_color.x());
                    double g_val = std::sqrt(pixel_color.y());
                    double b_val = std::sqrt(pixel_color.z());

                    // Clamp and convert to 8-bit
                    int ir = int(256 * std::clamp(r_val, 0.0, 0.999));
                    int ig = int(256 * std::clamp(g_val, 0.0, 0.999));
                    int ib = int(256 * std::clamp(b_val, 0.0, 0.999));

                    scanLine[i] = qRgb(ir, ig, ib);
                }
            }
        };

        // Launch threads
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(render_scanlines, t);
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }

    // Initialize ray tracer camera from GL camera parameters
    void initialize_from_gl_camera(const graphics::Cameraf& gl_camera, int image_width, int image_height) {
        // Extract parameters from graphics::Camera
        auto eye = gl_camera.getEye();
        auto lookat = gl_camera.getLookAt();
        auto up = gl_camera.getUp();
        double vfov = gl_camera.getFOV();

        point3 lookfrom(eye.x(), eye.y(), eye.z());
        point3 look_at(lookat.x(), lookat.y(), lookat.z());
        vec3 vup(up.x(), up.y(), up.z());

        center = lookfrom;

        // Determine viewport dimensions
        double theta = degrees_to_radians(vfov);
        double h = std::tan(theta / 2);
        double focus_dist = (lookfrom - look_at).length();
        double viewport_height = 2 * h * focus_dist;
        double viewport_width = viewport_height * (double(image_width) / image_height);

        // Calculate camera basis vectors
        w = unit_vector(lookfrom - look_at);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate viewport vectors
        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;  // -v because image Y is top-to-bottom

        // Pixel deltas
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Upper left pixel location
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    // Generate ray for pixel (i, j) with random offset for anti-aliasing
    ray get_ray(int i, int j) const {
        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    // Random offset in [-0.5, 0.5] square for anti-aliasing
    vec3 sample_square() const {
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }

    // Get sky color with optional sun (matches CUDA exactly)
    color get_sky_color(const vec3& direction) const {
        vec3 unit_direction = unit_vector(direction);
        double t = 0.5 * (unit_direction.y() + 1.0);
        color sky = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);

        // Check for distant lights (act like sun in sky)
        for (const auto& light : lights) {
            if (light.type == graphics::LightType::Distant) {
                vec3 sun_dir = unit_vector(vec3(light.direction.x(), light.direction.y(), light.direction.z()));
                double sun_dot = dot(unit_direction, sun_dir);
                double sun_intensity = std::pow(std::max(0.0, sun_dot), 512.0);  // Very sharp sun disk
                color light_color(light.color.getR(), light.color.getG(), light.color.getB());
                sky = sky + sun_intensity * light_color * 2.0;  // Fixed sun brightness
            }
        }

        return sky;
    }

    // Sample direct lighting (matches CUDA exactly)
    color sample_direct_light(const hit_record& rec) const {
        color direct(0, 0, 0);

        for (const auto& light : lights) {
            vec3 light_dir;
            double light_dist;
            color light_intensity;

            if (light.type == graphics::LightType::Point) {
                // Point light
                vec3 light_pos(light.position.x(), light.position.y(), light.position.z());
                vec3 to_light = light_pos - rec.p;
                light_dist = to_light.length();
                light_dir = to_light / light_dist;
                double falloff = 1.0 / (light_dist * light_dist + 0.1);
                color light_color(light.color.getR(), light.color.getG(), light.color.getB());
                light_intensity = light_color * (light.intensity * 0.1) * falloff;
            }
            else if (light.type == graphics::LightType::Distant) {
                // Distant light (sun) - use direction, no falloff
                // Note: direction stored as "from light" so we use it directly (matches CUDA)
                light_dir = unit_vector(vec3(light.direction.x(), light.direction.y(), light.direction.z()));
                light_dist = infinity;
                color light_color(light.color.getR(), light.color.getG(), light.color.getB());
                light_intensity = light_color * (light.intensity * 0.01);  // Match CUDA scaling
            }
            else if (light.type == graphics::LightType::Sphere) {
                // Sphere light - sample random point (simplified, no random for now)
                vec3 light_pos(light.position.x(), light.position.y(), light.position.z());
                vec3 to_center = light_pos - rec.p;
                double dist_to_center = to_center.length();
                light_dist = dist_to_center;
                light_dir = to_center / dist_to_center;
                double falloff = 1.0 / (dist_to_center * dist_to_center + 0.1);
                color light_color(light.color.getR(), light.color.getG(), light.color.getB());
                light_intensity = light_color * (light.intensity * 0.1) * falloff;
            }
            else {
                continue;
            }

            // Check if surface faces light
            double n_dot_l = dot(rec.normal, light_dir);
            if (n_dot_l <= 0.0) continue;

            // Shadow ray
            ray shadow_ray(rec.p + rec.normal * 0.001, light_dir);
            hit_record shadow_rec;
            bool in_shadow = world->hit(shadow_ray, interval(0.001, light_dist - 0.001), shadow_rec);

            if (!in_shadow) {
                direct = direct + light_intensity * n_dot_l;
            }
        }

        return direct;
    }

    // Ray color matching CUDA implementation exactly
    color ray_color(const ray& r, int depth) const {
        ray cur_ray = r;
        color cur_attenuation(1.0, 1.0, 1.0);
        color accumulated_light(0.0, 0.0, 0.0);

        for (int i = 0; i < depth; i++) {
            hit_record rec;
            if (world && world->hit(cur_ray, interval(0.001, infinity), rec)) {
                ray scattered;
                color attenuation;
                if (rec.mat->scatter(cur_ray, rec, attenuation, scattered)) {
                    // Sample direct lighting on all bounces for proper color bleeding
                    // The cur_attenuation naturally reduces contribution of later bounces
                    if (!lights.empty()) {
                        color direct = sample_direct_light(rec);
                        accumulated_light = accumulated_light + cur_attenuation * attenuation * direct;
                    }

                    cur_attenuation = cur_attenuation * attenuation;
                    cur_ray = scattered;
                } else {
                    return accumulated_light;
                }
            } else {
                // Hit sky
                color sky_color = get_sky_color(cur_ray.direction());
                return accumulated_light + cur_attenuation * sky_color;
            }
        }
        return accumulated_light;
    }
};

} // namespace rt

#endif
