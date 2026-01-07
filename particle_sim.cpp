#include "particle_sim.h"

Vector3 ParticleEnsemble::lorentzForce(
    const Vector3& v,
    const EMField& field
) const {
    return (field.E + cross(v, field.B) / LIGHT_VELOCITY) * ELECTRON_CHARGE;
}

void ParticleEnsemble::update(FP dt, const EMField& field) {
    const FP invMass = 1.0 / ELECTRON_MASS;

    for (auto& p : particles) {
        Vector3 a = lorentzForce(p.v, field) * invMass;

        p.r += p.v * dt;

        p.v += a * dt;
    }
}
