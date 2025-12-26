#pragma once
#include <vector>
#include <cstddef>
#include <riscv_vector.h>
#include "common_types.h"

struct alignas(64) EMFieldSoA {
    FP Ex = 0.0, Ey = 0.0, Ez = 0.0;
    FP Bx = 0.0, By = 0.0, Bz = 1.0;
};

struct alignas(64) ParticleSoA {
    std::vector<FP> rx, ry, rz;
    std::vector<FP> vx, vy, vz;

    explicit ParticleSoA(size_t n = 0) { resize(n); }

    void resize(size_t n) {
        rx.assign(n, 0.0);
        ry.assign(n, 0.0);
        rz.assign(n, 0.0);
        vx.assign(n, 1.0);
        vy.assign(n, 0.0);
        vz.assign(n, 0.0);
    }

    size_t size() const { return rx.size(); }
};

void updateSIMD(ParticleSoA& p, FP dt, const EMFieldSoA& field);