#include <iostream>
#include <random>

#include "particle_sim.h"

inline FP absDiff(FP a, FP b) {
    return std::abs(a - b);
}

inline FP maxComponentDiff(const Vector3& a, const Vector3& b) {
    FP dx = absDiff(a.x, b.x);
    FP dy = absDiff(a.y, b.y);
    FP dz = absDiff(a.z, b.z);

    return std::max(dx, std::max(dy, dz));
}

bool validateSIMD(
    ParticleEnsemble scalar,
    ParticleEnsemble simd,
    FP dt,
    const EMField& field,
    FP eps
) {
    if (scalar.particles.size() != simd.particles.size()) {
        std::cerr << "Size mismatch\n";
        return false;
    }

    scalar.update(dt, field);
    simd.updateSIMD(dt, field);

    FP maxErr = 0.0;

    for (size_t i = 0; i < scalar.particles.size(); ++i) {
        const Particle& a = scalar.particles[i];
        const Particle& b = simd.particles[i];

        FP dr = maxComponentDiff(a.r, b.r);
        FP dv = maxComponentDiff(a.v, b.v);

        maxErr = std::max(maxErr, std::max(dr, dv));

        if (dr > eps || dv > eps) {
            std::cerr << "Mismatch at i=" << i << "\n";
            std::cerr << "dr=" << dr << " dv=" << dv << "\n";
            return false;
        }
    }

    std::cout << "Validation OK, max error = " << maxErr << "\n";
    return true;
}

int main() {
    constexpr size_t N = 1024;
    constexpr FP dt = 1e-3;
    constexpr FP eps = 1e-7;

    ParticleEnsemble ref(N);
    ParticleEnsemble vec(N);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<FP> dist(-1.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        Vector3 r { dist(rng), dist(rng), dist(rng) };
        Vector3 v { dist(rng), dist(rng), dist(rng) };

        ref.particles[i].r = r;
        ref.particles[i].v = v;

        vec.particles[i] = ref.particles[i];
    }

    EMField field;
    field.E = {0.1, -0.2, 0.3};
    field.B = {0.0, 0.0, 1.0};

    bool ok = validateSIMD(ref, vec, dt, field, eps);

    return ok ? 0 : 1;
}