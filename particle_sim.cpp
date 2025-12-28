#include "particle_sim.h"

Vector3 ParticleEnsemble::lorentz_force(
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

        Vector3 k1v = lorentz_force(p.v, field);
        Vector3 k2v = lorentz_force(p.v + k1v * (0.5 * dt), field);
        Vector3 k3v = lorentz_force(p.v + k2v * (0.5 * dt), field);
        Vector3 k4v = lorentz_force(p.v + k3v * dt, field);

        p.v += (k1v + k2v*2 + k3v*2 + k4v) * coeffV;
    }
}
