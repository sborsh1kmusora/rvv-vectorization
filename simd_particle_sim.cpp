#include <riscv_vector.h>

#include "particle_sim.h"

void ParticleEnsemble::updateSIMD(FP dt, const EMField& field) {
    const size_t n = particles.size();
    if (n == 0) return;

    constexpr size_t stride = sizeof(Particle);

    const FP invMass = FP(1.0) / ELECTRON_MASS;
    const FP ae = ELECTRON_CHARGE * invMass;
    const FP k  = (ELECTRON_CHARGE / LIGHT_VELOCITY) * invMass;

    const FP Ex = field.E.x;
    const FP Ey = field.E.y;
    const FP Ez = field.E.z;

    const FP Bx = field.B.x;
    const FP By = field.B.y;
    const FP Bz = field.B.z;

    size_t i = 0;
    const size_t vlmax = __riscv_vsetvlmax_e64m1();

    for (; i + vlmax <= n; i += vlmax) {
        const size_t vl = vlmax;

        __builtin_prefetch(&particles[i + vl], 0, 3);

        FP* base = reinterpret_cast<FP*>(&particles[i]);

        FP* rxp = base + 0;
        FP* ryp = base + 1;
        FP* rzp = base + 2;
        FP* vxp = base + 3;
        FP* vyp = base + 4;
        FP* vzp = base + 5;

        vfloat64m1_t rx = __riscv_vlse64_v_f64m1(rxp, stride, vl);
        vfloat64m1_t ry = __riscv_vlse64_v_f64m1(ryp, stride, vl);
        vfloat64m1_t rz = __riscv_vlse64_v_f64m1(rzp, stride, vl);

        vfloat64m1_t vx = __riscv_vlse64_v_f64m1(vxp, stride, vl);
        vfloat64m1_t vy = __riscv_vlse64_v_f64m1(vyp, stride, vl);
        vfloat64m1_t vz = __riscv_vlse64_v_f64m1(vzp, stride, vl);

        vfloat64m1_t ax =
            __riscv_vfmacc_vf_f64m1(
                __riscv_vfmul_vf_f64m1(vy,  Bz * k, vl),
                -By * k,
                vz,
                vl
            );

        vfloat64m1_t ay =
            __riscv_vfmacc_vf_f64m1(
                __riscv_vfmul_vf_f64m1(vz,  Bx * k, vl),
                -Bz * k,
                vx,
                vl
            );

        vfloat64m1_t az =
            __riscv_vfmacc_vf_f64m1(
                __riscv_vfmul_vf_f64m1(vx,  By * k, vl),
                -Bx * k,
                vy,
                vl
            );

        ax = __riscv_vfadd_vf_f64m1(ax, Ex * ae, vl);
        ay = __riscv_vfadd_vf_f64m1(ay, Ey * ae, vl);
        az = __riscv_vfadd_vf_f64m1(az, Ez * ae, vl);

        rx = __riscv_vfmacc_vf_f64m1(rx, dt, vx, vl);
        ry = __riscv_vfmacc_vf_f64m1(ry, dt, vy, vl);
        rz = __riscv_vfmacc_vf_f64m1(rz, dt, vz, vl);

        vx = __riscv_vfmacc_vf_f64m1(vx, dt, ax, vl);
        vy = __riscv_vfmacc_vf_f64m1(vy, dt, ay, vl);
        vz = __riscv_vfmacc_vf_f64m1(vz, dt, az, vl);

        __riscv_vsse64_v_f64m1(rxp, stride, rx, vl);
        __riscv_vsse64_v_f64m1(ryp, stride, ry, vl);
        __riscv_vsse64_v_f64m1(rzp, stride, rz, vl);

        __riscv_vsse64_v_f64m1(vxp, stride, vx, vl);
        __riscv_vsse64_v_f64m1(vyp, stride, vy, vl);
        __riscv_vsse64_v_f64m1(vzp, stride, vz, vl);
    }

    if (i < n) {
        const size_t vl = __riscv_vsetvl_e64m1(n - i);

        FP* base = reinterpret_cast<FP*>(&particles[i]);

        FP* rxp = base + 0;
        FP* ryp = base + 1;
        FP* rzp = base + 2;
        FP* vxp = base + 3;
        FP* vyp = base + 4;
        FP* vzp = base + 5;

        vfloat64m1_t rx = __riscv_vlse64_v_f64m1(rxp, stride, vl);
        vfloat64m1_t ry = __riscv_vlse64_v_f64m1(ryp, stride, vl);
        vfloat64m1_t rz = __riscv_vlse64_v_f64m1(rzp, stride, vl);

        vfloat64m1_t vx = __riscv_vlse64_v_f64m1(vxp, stride, vl);
        vfloat64m1_t vy = __riscv_vlse64_v_f64m1(vyp, stride, vl);
        vfloat64m1_t vz = __riscv_vlse64_v_f64m1(vzp, stride, vl);

        vfloat64m1_t ax =
            __riscv_vfmacc_vf_f64m1(
                __riscv_vfmul_vf_f64m1(vy,  Bz * k, vl),
                -By * k,
                vz,
                vl
            );

        vfloat64m1_t ay =
            __riscv_vfmacc_vf_f64m1(
                __riscv_vfmul_vf_f64m1(vz,  Bx * k, vl),
                -Bz * k,
                vx,
                vl
            );

        vfloat64m1_t az =
            __riscv_vfmacc_vf_f64m1(
                __riscv_vfmul_vf_f64m1(vx,  By * k, vl),
                -Bx * k,
                vy,
                vl
            );

        ax = __riscv_vfadd_vf_f64m1(ax, Ex * ae, vl);
        ay = __riscv_vfadd_vf_f64m1(ay, Ey * ae, vl);
        az = __riscv_vfadd_vf_f64m1(az, Ez * ae, vl);

        rx = __riscv_vfmacc_vf_f64m1(rx, dt, vx, vl);
        ry = __riscv_vfmacc_vf_f64m1(ry, dt, vy, vl);
        rz = __riscv_vfmacc_vf_f64m1(rz, dt, vz, vl);

        vx = __riscv_vfmacc_vf_f64m1(vx, dt, ax, vl);
        vy = __riscv_vfmacc_vf_f64m1(vy, dt, ay, vl);
        vz = __riscv_vfmacc_vf_f64m1(vz, dt, az, vl);

        __riscv_vsse64_v_f64m1(rxp, stride, rx, vl);
        __riscv_vsse64_v_f64m1(ryp, stride, ry, vl);
        __riscv_vsse64_v_f64m1(rzp, stride, rz, vl);

        __riscv_vsse64_v_f64m1(vxp, stride, vx, vl);
        __riscv_vsse64_v_f64m1(vyp, stride, vy, vl);
        __riscv_vsse64_v_f64m1(vzp, stride, vz, vl);
    }
}
