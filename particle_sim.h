#pragma once

#include <vector>

#include "common_types.h"

inline Vector3 cross(const Vector3& a, const Vector3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

struct EMField {
    Vector3 E{0,0,0};
    Vector3 B{0,0,1};
};

class ParticleEnsemble {
public:
    explicit ParticleEnsemble(size_t n = 0) {
        particles.resize(n);
    }

    void update(FP dt, const EMField& field);

    std::vector<Particle> particles;

private:
    Vector3 lorentz_force(const Vector3& v, const EMField& field) const;
};
