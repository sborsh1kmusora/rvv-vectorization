#include "vector_particle_sim.h"
#include <riscv_vector.h>
#include <cstddef>
#include <vector>
#include "common_types.h"

static inline void lorentz_force(
    vfloat64m2_t vx, vfloat64m2_t vy, vfloat64m2_t vz,
    vfloat64m2_t& fx, vfloat64m2_t& fy, vfloat64m2_t& fz,
    const EMFieldSoA& field,
    size_t vl
) {
    const vfloat64m2_t vEx = __riscv_vfmv_v_f_f64m2(field.Ex, vl);
    const vfloat64m2_t vEy = __riscv_vfmv_v_f_f64m2(field.Ey, vl);
    const vfloat64m2_t vEz = __riscv_vfmv_v_f_f64m2(field.Ez, vl);
    const vfloat64m2_t vBx = __riscv_vfmv_v_f_f64m2(field.Bx, vl);
    const vfloat64m2_t vBy = __riscv_vfmv_v_f_f64m2(field.By, vl);
    const vfloat64m2_t vBz = __riscv_vfmv_v_f_f64m2(field.Bz, vl);

    const vfloat64m2_t inv_c = __riscv_vfmv_v_f_f64m2(1.0 / LIGHT_VELOCITY, vl);
    const vfloat64m2_t q = __riscv_vfmv_v_f_f64m2(ELECTRON_CHARGE, vl);

    fx = __riscv_vfadd_vv_f64m2(vEx,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfsub_vv_f64m2(
                    __riscv_vfmul_vv_f64m2(vy, vBz, vl),
                    __riscv_vfmul_vv_f64m2(vz, vBy, vl), vl),
                inv_c, vl), vl);

    fy = __riscv_vfadd_vv_f64m2(vEy,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfsub_vv_f64m2(
                    __riscv_vfmul_vv_f64m2(vz, vBx, vl),
                    __riscv_vfmul_vv_f64m2(vx, vBz, vl), vl),
                inv_c, vl), vl);

    fz = __riscv_vfadd_vv_f64m2(vEz,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfsub_vv_f64m2(
                    __riscv_vfmul_vv_f64m2(vx, vBy, vl),
                    __riscv_vfmul_vv_f64m2(vy, vBx, vl), vl),
                inv_c, vl), vl);

    fx = __riscv_vfmul_vv_f64m2(fx, q, vl);
    fy = __riscv_vfmul_vv_f64m2(fy, q, vl);
    fz = __riscv_vfmul_vv_f64m2(fz, q, vl);
}

void updateSIMD(ParticleSoA& p, FP dt, const EMFieldSoA& field) {
    size_t n = p.size();
    size_t i = 0;

    while (i < n) {
        size_t vl = __riscv_vsetvl_e64m2(n - i);

        alignas(64) vfloat64m2_t rx = __riscv_vle64_v_f64m2(&p.rx[i], vl);
        alignas(64) vfloat64m2_t ry = __riscv_vle64_v_f64m2(&p.ry[i], vl);
        alignas(64) vfloat64m2_t rz = __riscv_vle64_v_f64m2(&p.rz[i], vl);

        alignas(64) vfloat64m2_t vx = __riscv_vle64_v_f64m2(&p.vx[i], vl);
        alignas(64) vfloat64m2_t vy = __riscv_vle64_v_f64m2(&p.vy[i], vl);
        alignas(64) vfloat64m2_t vz = __riscv_vle64_v_f64m2(&p.vz[i], vl);

        vfloat64m2_t k1x, k1y, k1z;
        lorentz_force(vx, vy, vz, k1x, k1y, k1z, field, vl);

        const vfloat64m2_t half_dt = __riscv_vfmv_v_f_f64m2(0.5 * dt, vl);

        auto vx2 = __riscv_vfadd_vv_f64m2(vx, __riscv_vfmul_vv_f64m2(k1x, half_dt, vl), vl);
        auto vy2 = __riscv_vfadd_vv_f64m2(vy, __riscv_vfmul_vv_f64m2(k1y, half_dt, vl), vl);
        auto vz2 = __riscv_vfadd_vv_f64m2(vz, __riscv_vfmul_vv_f64m2(k1z, half_dt, vl), vl);

        vfloat64m2_t k2x, k2y, k2z;
        lorentz_force(vx2, vy2, vz2, k2x, k2y, k2z, field, vl);

        auto vx3 = __riscv_vfadd_vv_f64m2(vx, __riscv_vfmul_vv_f64m2(k2x, half_dt, vl), vl);
        auto vy3 = __riscv_vfadd_vv_f64m2(vy, __riscv_vfmul_vv_f64m2(k2y, half_dt, vl), vl);
        auto vz3 = __riscv_vfadd_vv_f64m2(vz, __riscv_vfmul_vv_f64m2(k2z, half_dt, vl), vl);

        vfloat64m2_t k3x, k3y, k3z;
        lorentz_force(vx3, vy3, vz3, k3x, k3y, k3z, field, vl);

        auto vx4 = __riscv_vfadd_vv_f64m2(vx, __riscv_vfmul_vv_f64m2(k3x, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        auto vy4 = __riscv_vfadd_vv_f64m2(vy, __riscv_vfmul_vv_f64m2(k3y, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        auto vz4 = __riscv_vfadd_vv_f64m2(vz, __riscv_vfmul_vv_f64m2(k3z, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);

        vfloat64m2_t k4x, k4y, k4z;
        lorentz_force(vx4, vy4, vz4, k4x, k4y, k4z, field, vl);

        const vfloat64m2_t dt6 = __riscv_vfmv_v_f_f64m2(dt / 6.0, vl);

        auto k2x2 = __riscv_vfmul_vf_f64m2(k2x, 2.0, vl);
        auto k3x2 = __riscv_vfmul_vf_f64m2(k3x, 2.0, vl);
        vx = __riscv_vfadd_vv_f64m2(vx, __riscv_vfmul_vv_f64m2(__riscv_vfadd_vv_f64m2(__riscv_vfadd_vv_f64m2(k1x, k2x2, vl),
                __riscv_vfadd_vv_f64m2(k3x2, k4x, vl), vl), dt6, vl), vl);

        auto k2y2 = __riscv_vfmul_vf_f64m2(k2y, 2.0, vl);
        auto k3y2 = __riscv_vfmul_vf_f64m2(k3y, 2.0, vl);
        vy = __riscv_vfadd_vv_f64m2(vy, __riscv_vfmul_vv_f64m2(__riscv_vfadd_vv_f64m2(__riscv_vfadd_vv_f64m2(k1y, k2y2, vl),
                __riscv_vfadd_vv_f64m2(k3y2, k4y, vl), vl), dt6, vl), vl);

        auto k2z2 = __riscv_vfmul_vf_f64m2(k2z, 2.0, vl);
        auto k3z2 = __riscv_vfmul_vf_f64m2(k3z, 2.0, vl);
        vz = __riscv_vfadd_vv_f64m2(vz, __riscv_vfmul_vv_f64m2(__riscv_vfadd_vv_f64m2(__riscv_vfadd_vv_f64m2(k1z, k2z2, vl),
                __riscv_vfadd_vv_f64m2(k3z2, k4z, vl), vl), dt6, vl), vl);

        rx = __riscv_vfadd_vv_f64m2(rx, __riscv_vfmul_vv_f64m2(vx, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        ry = __riscv_vfadd_vv_f64m2(ry, __riscv_vfmul_vv_f64m2(vy, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        rz = __riscv_vfadd_vv_f64m2(rz, __riscv_vfmul_vv_f64m2(vz, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);

        __riscv_vse64_v_f64m2(&p.rx[i], rx, vl);
        __riscv_vse64_v_f64m2(&p.ry[i], ry, vl);
        __riscv_vse64_v_f64m2(&p.rz[i], rz, vl);
        __riscv_vse64_v_f64m2(&p.vx[i], vx, vl);
        __riscv_vse64_v_f64m2(&p.vy[i], vy, vl);
        __riscv_vse64_v_f64m2(&p.vz[i], vz, vl);

        i += vl;
    }
}
