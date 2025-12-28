#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "particle_sim.h"
#include "soa_particle_sim.h"

using FP = double;

struct ErrorStats {
    FP max_abs = 0.0;
    FP rms = 0.0;
};

ErrorStats compare(const std::vector<FP>& a, const std::vector<FP>& b)
{
    ErrorStats e;
    FP sum2 = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        FP d = std::abs(a[i] - b[i]);
        if (d > e.max_abs) e.max_abs = d;
        sum2 += d * d;
    }

    e.rms = std::sqrt(sum2 / a.size());
    return e;
}

void print(const char* name, const ErrorStats& e) {
    std::cout << name
              << "  max=" << e.max_abs
              << "  rms=" << e.rms << '\n';
}

int main() {
    constexpr size_t N = 5000;
    constexpr FP dt = 1e-9;
    constexpr int steps = 500;

    EMField field_scalar;
    EMFieldSoA field_simd;

    ParticleEnsemble scalar(N);
    ParticleSoA simd(N);

    for (size_t i = 0; i < N; ++i) {
        scalar.particles[i].r = {0,0,0};
        scalar.particles[i].v = {1,0,0};

        simd.rx[i] = 0; simd.ry[i] = 0; simd.rz[i] = 0;
        simd.vx[i] = 1; simd.vy[i] = 0; simd.vz[i] = 0;
    }

    for (int s = 0; s < steps; ++s) {
        scalar.update(dt, field_scalar);
        update(simd, dt, field_simd);
    }

    std::vector<FP> rx(N), ry(N), rz(N);
    std::vector<FP> vx(N), vy(N), vz(N);

    for (size_t i = 0; i < N; ++i) {
        rx[i] = scalar.particles[i].r.x;
        ry[i] = scalar.particles[i].r.y;
        rz[i] = scalar.particles[i].r.z;

        vx[i] = scalar.particles[i].v.x;
        vy[i] = scalar.particles[i].v.y;
        vz[i] = scalar.particles[i].v.z;
    }

    print("rx", compare(rx, simd.rx));
    print("ry", compare(ry, simd.ry));
    print("rz", compare(rz, simd.rz));
    print("vx", compare(vx, simd.vx));
    print("vy", compare(vy, simd.vy));
    print("vz", compare(vz, simd.vz));

    return 0;
}