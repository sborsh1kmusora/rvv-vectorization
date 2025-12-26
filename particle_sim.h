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
    Vector3 lorentzForce(const Vector3& v, const EMField& field) const;
};

Vector3 ParticleEnsemble::lorentzForce(
    const Vector3& v,
    const EMField& field
) const {
    return (field.E + cross(v, field.B) / LIGHT_VELOCITY) * ELECTRON_CHARGE;
}

void ParticleEnsemble::update(FP dt, const EMField& field) {
    const FP dt6 = dt / 6.0;
    const FP coeffV = dt6 / ELECTRON_MASS;

    for (auto& p : particles) {
        Vector3 k1r = p.v;
        Vector3 k2r = p.v + k1r * (0.5 * dt);
        Vector3 k3r = p.v + k2r * (0.5 * dt);
        Vector3 k4r = p.v + k3r * dt;

        p.r += (k1r + k2r*2 + k3r*2 + k4r) * dt6;

        Vector3 k1v = lorentzForce(p.v, field);
        Vector3 k2v = lorentzForce(p.v + k1v * (0.5 * dt), field);
        Vector3 k3v = lorentzForce(p.v + k2v * (0.5 * dt), field);
        Vector3 k4v = lorentzForce(p.v + k3v * dt, field);

        p.v += (k1v + k2v*2 + k3v*2 + k4v) * coeffV;
    }
}