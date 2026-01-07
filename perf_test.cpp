#include <iostream>
#include <chrono>
#include <random>

#include "particle_sim.h"

void initParticles(ParticleEnsemble& p) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<FP> dist(-1.0, 1.0);

    for (auto& part : p.particles) {
        part.r = { dist(rng), dist(rng), dist(rng) };
        part.v = { dist(rng), dist(rng), dist(rng) };
    }
}

template <typename Func>
double benchmark(
    ParticleEnsemble particles,
    Func stepFunc,
    FP dt,
    const EMField& field,
    size_t steps
) {
    for (size_t i = 0; i < 100; ++i)
        stepFunc(particles, dt, field);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < steps; ++i)
        stepFunc(particles, dt, field);

    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t1 - t0;
    return elapsed.count() / steps;
}

// m1 
// **** REAL SIMULATION ****
// Particles      : 10000
// Steps          : 1000
// Scalar time    : 716.34 us / step
// SIMD time      : 722.749 us / step
// Speedup        : 0.991132x
// m2 
// SIMD time      : 654.226 us / step
// Speedup        : 1.09495x
// m4 
// SIMD time      : 640.201 us / step
// Speedup        : 1.11893x
int main() {
    constexpr size_t N = 10000;   
    constexpr size_t steps = 1000;
    constexpr FP dt = 1e-9;

    ParticleEnsemble scalar(N);
    ParticleEnsemble simd(N);

    initParticles(scalar);
    simd = scalar;

    EMField field;
    field.E = {0.1, -0.2, 0.3};
    field.B = {0.0, 0.0, 1.0};

    double tScalar = benchmark(
        scalar,
        [](ParticleEnsemble& p, FP dt, const EMField& f) {
            p.update(dt, f);
        },
        dt, field, steps
    );

    double tSIMD = benchmark(
        simd,
        [](ParticleEnsemble& p, FP dt, const EMField& f) {
            p.updateSIMD(dt, f);
        },
        dt, field, steps
    );

    std::cout << "Particles      : " << N << "\n";
    std::cout << "Steps          : " << steps << "\n";
    std::cout << "Scalar time    : " << tScalar * 1e6 << " us / step\n";
    std::cout << "SIMD time      : " << tSIMD   * 1e6 << " us / step\n";
    std::cout << "Speedup        : " << tScalar / tSIMD << "x\n";

    return 0;
}
