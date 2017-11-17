typedef struct {
  float3 ori;
  float3 dir;
} Ray;

typedef struct {
  float3 cen;
  float rad;
  float3 col;
  float3 emi;
} Sphere;

typedef struct {
  float3 ip;
  float3 nor;
  float3 col;
  float3 emi;
} Hit;

__constant Sphere scene[] = {
  {(float3)(-2.5, -3.0, 0.0), 2.0, (float3)(1.0, 1.0, 1.0)},
  {(float3)(2.5, -3.0, 2.0), 2.0, (float3)(1.0, 1.0, 1.0)},
  {(float3)(0.0, 194.99, 0.0), 190.0, (float3)(1.0, 1.0, 1.0), (float3)(20, 20, 20)},
  {(float3)(0.0, -10005.0, 0.0), 10000.0, (float3)(0.7, 0.7, 0.7)},
  {(float3)(0.0, 10005.0, 0.0), 10000.0, (float3)(0.7, 0.7, 0.7)},
  {(float3)(-10008.0, 0.0, 0.0), 10000.0, (float3)(1.0, 0.0, 1.0)},
  {(float3)(10008.0, 0.0, 0.0), 10000.0, (float3)(0.0, 1.0, 1.0)},
  {(float3)(0.0, 0.0, -10005.0), 10000.0, (float3)(0.7, 0.7, 0.7)},
  {(float3)(0.0, 10.0, -10005.0), 10000.0, (float3)(0.7, 0.7, 0.7)}
};

Ray create_ray(float u, float v) {
  float3 on_plane = {u - 0.5, v - 0.5, 20};
  float3 eye = {0, 0, 21};

  return (Ray){eye, normalize(on_plane - eye)};
}

float intersect(Ray* ray, __constant Sphere* sphere) {
  float3 op = sphere->cen - ray->ori;
  float eps = 1e-4;
  float b = dot(op, ray->dir);
  float det2 = b * b - dot(op, op) + sphere->rad * sphere->rad;
  if (det2 < 0) return INFINITY;

  float det = sqrt(det2);
  float t;

  return (t=b-det) > eps ? t : ((t=b+det)>eps ? t : INFINITY);
}

bool intersect_scene(Ray* ray, Hit* hit) {
  int scene_size = sizeof(scene) / sizeof(Sphere);

  bool found = false;
  float tmax = INFINITY;
  int id = 0;
  for (int i = 0; i < scene_size; i++) {
    float t = intersect(ray, &scene[i]);

    if (t < tmax) {
      found = true;
      tmax = t;
      id = i;
    }
  }

  if (found) {
    float3 ip = ray->ori + ray->dir * tmax;

    hit->ip = ip;
    hit->nor = normalize(ip - scene[id].cen);
    hit->col = scene[id].col;
    hit->emi = scene[id].emi;

    return true;
  } else {
    return false;
  }
}

float prng(unsigned int* seed) {
  *seed = (*seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  return *seed / 4294967295.0f;
}

float3 direction(float3 nor, unsigned int* seed) {
  float r1 = 2.0f * M_PI_F * prng(seed);
  float r2 = prng(seed);
  float r2s = sqrt(r2);

  float3 u = normalize(cross(fabs(nor.x) > 0.1 ? (float3)(0.0, 1.0, 0.0) : (float3)(1.0, 0.0, 0.0), nor));
  float3 v = cross(nor, u);
  return normalize((u*cos(r1)*r2s + v*sin(r1)*r2s + nor*sqrt(1.0f - r2)));
}
// double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2);
//      Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u;
//      Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
//      return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));

__kernel void trace(uint2 size, __global const float* seeds, __write_only image2d_t output) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  unsigned int seed = (unsigned int)(seeds[pos.y * size.x + pos.x] * 4294967295.0f);

  float3 total = (float3)(0.0f, 0.0f, 0.0f);

  for (int j = 0; j < 2000; j++) {
    Ray ray = create_ray(pos.x / (float)size.x, pos.y / (float)size.y);

    float3 radiance = (float3)(0.0f, 0.0f, 0.0f);
    float3 weight = (float3)(1, 1, 1);

    for (int i = 0; i < 10; i++) {
      Hit hit;
      if (!intersect_scene(&ray, &hit)) break;

      radiance += weight * hit.emi;

      weight *= hit.col;

      ray = (Ray){hit.ip + hit.nor * 0.01f, direction(hit.nor, &seed)};
    }

    total += radiance;
  }

  total /= 2000.0f;

  write_imagef(output, pos, (float4)(clamp(total, 0.0f, 1.0f), 1.0f));
}
