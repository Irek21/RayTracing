#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <fstream>
#include <algorithm>
#include "customvectors.h"
#include <iostream>
#include <cstdint>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;

const uint32_t RED   = 0x000000FF;
const uint32_t GREEN = 0x0000FF00;
const uint32_t BLUE  = 0x00FF0000;

#include <string>
#include <vector>
#include <unordered_map>

struct Light {
    Light(const float3 &p, const float i) : position(p), intensity(i) {}
    float3 position;
    float intensity;
};

struct Material {
    Material(float r, float2 f, float refl, float refr, float3 color, float spec) : refr_idx(r), Fong_intense(f), reflection(refl), refraction(refr), color(color), specular_exponent(spec) {}
    Material(): Fong_intense(1, 0), color(), specular_exponent() {}
    float refr_idx;
    float2 Fong_intense;
    float refraction;
    float reflection;
    float3 color;
    float specular_exponent;
};

struct Point {
  float3 N;
  Material material;
  float3 coords;
  bool exists;
};

struct Ray {
  Ray(float3 st, float3 dir): start(st), dir(dir) {}
  float3 start;
  float3 dir;
};

struct Sphere {
    float3 center;
    float radius;
    Material material;
    Sphere(const float3 &c, const float r, const Material &m) : center(c), radius(r), material(m) {}
    bool ray_intersection(Ray ray, float &t0) const {
        float3 L = center - ray.start;
        float tca = L * ray.dir;
        float d2 = L * L - tca * tca;
        if (d2 > radius * radius) return false;
        float thc = sqrtf(radius * radius - d2);
        t0 = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }
};

float3 reflect(float3 I, float3 N) {
    return N * 2.f * (I * N) - I;
}

float3 refract(float3 I, float3 N, const float eta_t, const float eta_i = 1.f) {
    float cosi = -max(-1.f, min(1.f, I * N));
    if (cosi < 0) return refract(I, -N, eta_i, eta_t);
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? float3(1, 0, 0) : I * eta + N * (eta * cosi - sqrtf(k));
}

Point ray_scene_intersection(Ray ray, const vector<Sphere> &spheres_vao) {
  Point hit;
  float sph_dist = std::numeric_limits<float>::max();
  for (auto& sphere : spheres_vao) {
    float distance;
    if (sphere.ray_intersection(ray, distance) && (distance < sph_dist)) {
      sph_dist = distance;
      hit.coords = ray.start + ray.dir * distance;
      hit.material = sphere.material;
      hit.N = (hit.coords - sphere.center).normalize();
    }
  }
  if (sph_dist < 1000) hit.exists = true;
  else hit.exists = false;

  float chessboard_dist = std::numeric_limits<float>::max();
  if (fabs(ray.dir.y) > 1e-3) {
    float d = -(ray.start.y + 8) / ray.dir.y;
    float3 pt = ray.start + ray.dir * d;
    if (d > 0 && fabs(pt.x) < 10 && pt.z < -10 && pt.z > -30 && d < sph_dist) {
      chessboard_dist = d;
      hit.exists = true;
      hit.coords = pt;
      hit.N = float3(0, 1, 0);
      hit.material.color = (int(.5 * hit.coords.x + 1000) + int(.5 * hit.coords.z)) & 1 ? float3(.3, .3, .3) : float3(.0, .0, .1);
    }
  }
  return hit;
}

class FragmentShader {
public:
  FragmentShader(int w, int h, vector<Sphere> &sph_vao, vector<Light> &lights, vector<float3> &map, float3 amb_light=float3(0, 0, 0)):
      width(w), height(h), spheres_vao(sph_vao), lights(lights), envmap(map), amb_light(amb_light) {}
  int width;
  int height;
  vector<Sphere> spheres_vao;
  vector<Light> lights;
  vector<float3> envmap;
  float3 amb_light;
  float3 trace_ray(Ray ray, size_t depth);
  vector<unsigned char> render();
};

float3 FragmentShader::trace_ray(Ray ray, size_t depth=0) {
  if (depth > 4) {
    if (ray.dir.z < 0) {
      int map_i = (int) (900 * (ray.dir.x / (ray.dir.z)) + 1500 / 2);
      int map_j = (int) (900 * (ray.dir.y / (ray.dir.z)) + 1000 / 2);
      if ((map_i < 1500) && (map_j < 1000) && (map_i >= 0) && (map_j >= 0)) return envmap[map_j * 1500 + map_i];
    }
    return float3(0., 0., 0.);
  }
  Point hit = ray_scene_intersection(ray, spheres_vao);
  if (!hit.exists) {
    if (ray.dir.z < 0) {
      int map_i = (int) (900 * (ray.dir.x / (ray.dir.z)) + 1500 / 2);
      int map_j = (int) (900 * (ray.dir.y / (ray.dir.z)) + 1000 / 2);
      if ((map_i < 1500) && (map_j < 1000) && (map_i >= 0) && (map_j >= 0)) return envmap[map_j * 1500 + map_i];
    }
    return float3(0., 0., 0.);
  }
  float3 N = hit.N;
  Material material = hit.material;
  float3 coords = hit.coords;

  float diffuse_light_intensity = 0, specular_light_intensity = 0;
  for (size_t i = 0; i < lights.size(); ++i) {
    float3 dir_light = (lights[i].position - coords).normalize();
    float3 shadow_orig = dir_light * N < 0 ? coords - N * 1e-3 : coords + N * 1e-3;
    Ray ray_to_light(shadow_orig, dir_light);
    float distance_light = (lights[i].position - coords).norm();

    Point hit_scene = ray_scene_intersection(ray_to_light, spheres_vao);
    if (!(hit_scene.exists && ((hit_scene.coords - shadow_orig).norm() < distance_light))) {
      diffuse_light_intensity  += lights[i].intensity * std::max(0.f, dir_light * N);
      specular_light_intensity += powf(std::max(0.f, -reflect(dir_light, N) * ray.dir), material.specular_exponent) * lights[i].intensity;
    }
  }
  float3 pixel_RGB = amb_light + material.color * (diffuse_light_intensity * material.Fong_intense[0] + specular_light_intensity * material.Fong_intense[1]);

  if (material.reflection > 0) {
    float3 reflect_dir = -reflect(ray.dir, N).normalize();
    float3 reflect_orig = reflect_dir * N < 0 ? coords - N * 1e-3 : coords + N * 1e-3;
    Ray reflect_ray(reflect_orig, reflect_dir);
    pixel_RGB = pixel_RGB + trace_ray(reflect_ray, depth + 1) * material.reflection;
  }
  if (material.refraction > 0) {
    float3 refract_dir = refract(ray.dir, N, material.refr_idx).normalize();
    float3 refract_orig = refract_dir * N < 0 ? coords - N * 1e-3 : coords + N * 1e-3;
    Ray refract_ray(refract_orig, refract_dir);
    pixel_RGB = pixel_RGB + trace_ray(refract_ray, depth + 1) * material.refraction;
  }
  return pixel_RGB;
}

vector<unsigned char> FragmentShader::render() {
    const float fov      = M_PI/3.;
    float dir_z = -height / (2. * tan(fov / 2.));
    vector<float3> framebuffer(width * height);
    vector<unsigned char> image(width * height * 3);

    for (size_t j = 0; j < height; j++) {
      for (size_t i = 0; i < width; i++) {
        float dir_x =  (i + 0.5) -  width / 2.;
        float dir_y =  (j + 0.5) - height / 2.;
        framebuffer[i + j * width] = trace_ray(Ray(float3(0, 0, 0), float3(dir_x, -dir_y, dir_z).normalize()));
      }
    }

    for (size_t i = 0; i < height * width; ++i) {
        float3 &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max > 1) c = c * (1. / max);
        for (size_t j = 0; j < 3; j++) {
             image[3 * i + j] = (unsigned char) (255 * std::max(0.f, std::min(1.f, c[j])));
        }
    }
    return image;
}

int main(int argc, const char** argv)
{
  unordered_map<string, string> cmdLineParams;

  for(int i = 0; i < argc; i++)
  {
    string key(argv[i]);
    if(key.size() > 0 && key[0]=='-')
    {
      if(i != argc - 1) // not last argument
      {
        cmdLineParams[key] = argv[i + 1];
        i++;
      }
      else cmdLineParams[key] = "";
    }
  }

  string outFilePath = "zout.bmp";
  if (cmdLineParams.find("-out") != cmdLineParams.end())
    outFilePath = cmdLineParams["-out"];

  int sceneId = 0;
  if (cmdLineParams.find("-scene") != cmdLineParams.end())
    sceneId = atoi(cmdLineParams["-scene"].c_str());

  uint32_t color = 0;
  if (sceneId == 1)
    color = RED;
  else if (sceneId == 2)
    color = RED | GREEN;
  else if (sceneId == 3)
    color = BLUE;


  int envmap_width = 1500;
  int envmap_height = 1000;
  int n = -1;
  unsigned char *pixmap = stbi_load("../carpet.jpg", &envmap_width, &envmap_height, &n, 0);
  if (!pixmap || 3 != n) {
      cerr << "Error: can not load the environment map" << std::endl;
      return -1;
  }
  vector<float3> envmap(envmap_width * envmap_height);
  for (int j = envmap_height - 1; j >= 0 ; j--) {
      for (int i = 0; i < envmap_width; i++) {
          envmap[i + j * envmap_width] = float3(pixmap[(i + j * envmap_width) * 3 + 0], pixmap[(i + j * envmap_width) * 3 + 1], pixmap[(i + j * envmap_width) * 3 + 2]) * (1 / 255.);
      }
  }
  stbi_image_free(pixmap);

  int WIDTH = 1920;
  int HEIGHT = 1080;
  Material ivory(1.0, float2(0.6,  0.3), 0.1, 0.0, float3(0.4, 0.4, 0.3),   50.);
  Material glass(1.5, float2(0.0,  0.5), 0.1, 0.8, float3(0.6, 0.7, 0.8),  125.);
  Material matt(1.0, float2(0.8,  0.2), 0.0, 0.0, float3(0.1, 0.1, 0.4),   10.);
  Material mirror(1.0, float2(0.0, 10.0), 0.8, 0.0, float3(1.0, 1.0, 1.0), 1425.);

  vector<Sphere> spheres_vao;
  spheres_vao.push_back(Sphere(float3(-4, -2, -16), 3, matt));
  spheres_vao.push_back(Sphere(float3(10, 1, -18), 3, glass));
  spheres_vao.push_back(Sphere(float3(-10, 1, -18), 3, mirror));
  spheres_vao.push_back(Sphere(float3(2.5, -1, -10), 2, ivory));

  vector<Light>  lights;
  lights.push_back(Light(float3(-20, 20,  20), 1.5));
  lights.push_back(Light(float3( 30, 50, -25), 1.8));
  lights.push_back(Light(float3( 30, 20,  30), 1.7));
  FragmentShader FShader(WIDTH, HEIGHT, spheres_vao, lights, envmap);

  vector<unsigned char> image = FShader.render();

  stbi_write_bmp(outFilePath.c_str(), WIDTH, HEIGHT, 3, image.data());

  cout << "end." << endl;
  return 0;
}
