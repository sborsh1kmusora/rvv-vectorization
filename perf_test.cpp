#include <iostream>
#include <chrono>
#include <vector>

#include "particle_sim.h"
#include "simd_particle_sim.h"
#include "soa_particle_sim.h"

constexpr size_t N = 10000;
constexpr int STEPS = 100;
constexpr FP DT = 1e-9;

double test_scalar() {
    ParticleEnsemble ensemble(N);
    EMField field{{1.0, 0.5, 0.0}, {0.0, 0.0, 1.0}};

    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < STEPS; ++s) {
        ensemble.update(DT, field);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double test_simd_aos() {
    std::vector<Particle> particles(N);
    for (size_t i = 0; i < N; ++i) {
        particles[i].r = {FP(i)*0.01, FP(i)*0.01, FP(i)*0.01};
        particles[i].v = {0.0, 0.0, 0.0};
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < STEPS; ++s) {
        update(particles, DT);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

double test_simd_soa() {
    ParticleSoA particles(N);
    for (size_t i = 0; i < N; ++i) {
        particles.rx[i] = particles.ry[i] = particles.rz[i] = FP(i) * 0.01;
        particles.vx[i] = particles.vy[i] = particles.vz[i] = 0.0;
    }

    EMFieldSoA field{1.0, 0.5, 0.0, 0.0, 0.0, 1.0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < STEPS; ++s) {
        update(particles, DT, field);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    double t_scalar   = test_scalar();
    double t_simd_aos = test_simd_aos();
    double t_simd_soa = test_simd_soa();

    std::cout << "Scalar (AoS):        " << t_scalar   << " s\n";
    std::cout << "SIMD (AoS):          " << t_simd_aos << " s\n";
    std::cout << "SIMD + SoA (RVV):    " << t_simd_soa << " s\n";

    std::cout << "\nSpeedup SIMD vs scalar:      "
              << t_scalar / t_simd_aos << "x\n";
    std::cout << "Speedup SIMD+SoA vs scalar:  "
              << t_scalar / t_simd_soa << "x\n";

    return 0;
}
