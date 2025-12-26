#include <vector>
#include <riscv_vector.h>

#include "common_types.h"

extern Vector3 E;
extern Vector3 B;

inline void lorentz_force(
    vfloat64m1_t vx,
    vfloat64m1_t vy,
    vfloat64m1_t vz,
    vfloat64m1_t& fx,
    vfloat64m1_t& fy,
    vfloat64m1_t& fz,
    size_t vl
);

void updateSIMD(std::vector<Particle>& particles, FP dt);
