#include <riscv_vector.h>
#include <vector>
#include <cstddef>

#include "common_types.h"
#include "simd_particle_sim.h"

Vector3 E{0.0, 0.0, 0.0};
Vector3 B{0.0, 0.0, 1.0};

inline void lorentz_force(
    vfloat64m2_t vx,
    vfloat64m2_t vy,
    vfloat64m2_t vz,
    vfloat64m2_t& fx,
    vfloat64m2_t& fy,
    vfloat64m2_t& fz,
    size_t vl
) {
    // Загрузка электрического поля
    const vfloat64m2_t Ex = __riscv_vfmv_v_f_f64m2(E.x, vl);
    const vfloat64m2_t Ey = __riscv_vfmv_v_f_f64m2(E.y, vl);
    const vfloat64m2_t Ez = __riscv_vfmv_v_f_f64m2(E.z, vl);

    // Загрузка магнитного поля
    const vfloat64m2_t Bx = __riscv_vfmv_v_f_f64m2(B.x, vl);
    const vfloat64m2_t By = __riscv_vfmv_v_f_f64m2(B.y, vl);
    const vfloat64m2_t Bz = __riscv_vfmv_v_f_f64m2(B.z, vl);

    // Обратная скорость света
    const vfloat64m2_t inv_c = __riscv_vfmv_v_f_f64m2(1.0 / LIGHT_VELOCITY, vl);

    // Заряд электрона
    const vfloat64m2_t q = __riscv_vfmv_v_f_f64m2(ELECTRON_CHARGE, vl);

    // Ex + (vy*Bz-vz*By) / c
    fx = __riscv_vfadd_vv_f64m2(
            Ex,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfsub_vv_f64m2(
                    __riscv_vfmul_vv_f64m2(vy, Bz, vl),
                    __riscv_vfmul_vv_f64m2(vz, By, vl),
                    vl),
                inv_c,
                vl),
            vl);

    // Ey + (vz*Bx-vx*Bz) / c
    fy = __riscv_vfadd_vv_f64m2(
            Ey,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfsub_vv_f64m2(
                    __riscv_vfmul_vv_f64m2(vz, Bx, vl),
                    __riscv_vfmul_vv_f64m2(vx, Bz, vl),
                    vl),
                inv_c,
                vl),
            vl);

    // Ez + (vx*By-vy*Bx) / c
    fz = __riscv_vfadd_vv_f64m2(
            Ez,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfsub_vv_f64m2(
                    __riscv_vfmul_vv_f64m2(vx, By, vl),
                    __riscv_vfmul_vv_f64m2(vy, Bx, vl),
                    vl),
                inv_c,
                vl),
            vl);

    fx = __riscv_vfmul_vv_f64m2(fx, q, vl);
    fy = __riscv_vfmul_vv_f64m2(fy, q, vl);
    fz = __riscv_vfmul_vv_f64m2(fz, q, vl);
}

void update(std::vector<Particle>& particles, FP dt) {
    const size_t n = particles.size();
    size_t i = 0;

    while (i < n) {
        // Количество элементов, которые обрабатываются в одной SIMD инструкции
        size_t vl = __riscv_vsetvl_e64m2(n - i);

        // Загружает vl координат из памяти в векторный регистр
        alignas(64) vfloat64m2_t rx = __riscv_vle64_v_f64m2(&particles[i].r.x, vl);
        alignas(64) vfloat64m2_t ry = __riscv_vle64_v_f64m2(&particles[i].r.y, vl);
        alignas(64) vfloat64m2_t rz = __riscv_vle64_v_f64m2(&particles[i].r.z, vl);

        alignas(64) vfloat64m2_t vx = __riscv_vle64_v_f64m2(&particles[i].v.x, vl);
        alignas(64) vfloat64m2_t vy = __riscv_vle64_v_f64m2(&particles[i].v.y, vl);
        alignas(64) vfloat64m2_t vz = __riscv_vle64_v_f64m2(&particles[i].v.z, vl);

        // Сила Лоренца в начале шага (k1)
        vfloat64m2_t k1x, k1y, k1z;
        lorentz_force(vx, vy, vz, k1x, k1y, k1z, vl);

        // Вектор с константой 0.5 * dt для всех vl частиц
        const vfloat64m2_t half_dt = __riscv_vfmv_v_f_f64m2(0.5 * dt, vl);

        // Вычисление скорости частиц в середине шага: vx[i]+k1x[i]*dt/2
        auto vx2 = __riscv_vfadd_vv_f64m2(vx, __riscv_vfmul_vv_f64m2(k1x, half_dt, vl), vl);
        auto vy2 = __riscv_vfadd_vv_f64m2(vy, __riscv_vfmul_vv_f64m2(k1y, half_dt, vl), vl);
        auto vz2 = __riscv_vfadd_vv_f64m2(vz, __riscv_vfmul_vv_f64m2(k1z, half_dt, vl), vl);

        vfloat64m2_t k2x, k2y, k2z;
        lorentz_force(vx2, vy2, vz2, k2x, k2y, k2z, vl);

        auto vx3 = __riscv_vfadd_vv_f64m2(vx, __riscv_vfmul_vv_f64m2(k2x, half_dt, vl), vl);
        auto vy3 = __riscv_vfadd_vv_f64m2(vy, __riscv_vfmul_vv_f64m2(k2y, half_dt, vl), vl);
        auto vz3 = __riscv_vfadd_vv_f64m2(vz, __riscv_vfmul_vv_f64m2(k2z, half_dt, vl), vl);

        vfloat64m2_t k3x, k3y, k3z;
        lorentz_force(vx3, vy3, vz3, k3x, k3y, k3z, vl);

        auto vx4 = __riscv_vfadd_vv_f64m2(vx, __riscv_vfmul_vv_f64m2(k3x, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        auto vy4 = __riscv_vfadd_vv_f64m2(vy, __riscv_vfmul_vv_f64m2(k3y, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        auto vz4 = __riscv_vfadd_vv_f64m2(vz, __riscv_vfmul_vv_f64m2(k3z, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);

        vfloat64m2_t k4x, k4y, k4z;
        lorentz_force(vx4, vy4, vz4, k4x, k4y, k4z, vl);

        const vfloat64m2_t dt6 = __riscv_vfmv_v_f_f64m2(dt / 6.0, vl);

        // v = v + dt/6*(k1+2k2+3k3+k4)
        vx = __riscv_vfadd_vv_f64m2(vx,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfadd_vv_f64m2(
                    __riscv_vfadd_vv_f64m2(k1x, __riscv_vfmul_vf_f64m2(k2x, 2.0, vl), vl),
                    __riscv_vfadd_vv_f64m2(__riscv_vfmul_vf_f64m2(k3x, 2.0, vl), k4x, vl),
                    vl),
                dt6, vl), vl);

        vy = __riscv_vfadd_vv_f64m2(vy,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfadd_vv_f64m2(
                    __riscv_vfadd_vv_f64m2(k1y, __riscv_vfmul_vf_f64m2(k2y, 2.0, vl), vl),
                    __riscv_vfadd_vv_f64m2(__riscv_vfmul_vf_f64m2(k3y, 2.0, vl), k4y, vl),
                    vl),
                dt6, vl), vl);

        vz = __riscv_vfadd_vv_f64m2(vz,
            __riscv_vfmul_vv_f64m2(
                __riscv_vfadd_vv_f64m2(
                    __riscv_vfadd_vv_f64m2(k1z, __riscv_vfmul_vf_f64m2(k2z, 2.0, vl), vl),
                    __riscv_vfadd_vv_f64m2(__riscv_vfmul_vf_f64m2(k3z, 2.0, vl), k4z, vl),
                    vl),
                dt6, vl), vl);

        // r[i] = r[i] + v[i]*dt
        rx = __riscv_vfadd_vv_f64m2(rx, __riscv_vfmul_vv_f64m2(vx, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        ry = __riscv_vfadd_vv_f64m2(ry, __riscv_vfmul_vv_f64m2(vy, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);
        rz = __riscv_vfadd_vv_f64m2(rz, __riscv_vfmul_vv_f64m2(vz, __riscv_vfmv_v_f_f64m2(dt, vl), vl), vl);

        // Cохраняем обратно vl элементов из векторных регистров в массив частиц
        __riscv_vse64_v_f64m2(&particles[i].r.x, rx, vl);
        __riscv_vse64_v_f64m2(&particles[i].r.y, ry, vl);
        __riscv_vse64_v_f64m2(&particles[i].r.z, rz, vl);
        __riscv_vse64_v_f64m2(&particles[i].v.x, vx, vl);
        __riscv_vse64_v_f64m2(&particles[i].v.y, vy, vl);
        __riscv_vse64_v_f64m2(&particles[i].v.z, vz, vl);

        // Инкрементируем индекс на число обработанных элементов
        i += vl;
    }
}