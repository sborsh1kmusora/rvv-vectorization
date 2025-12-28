#pragma once

using FP = double;

struct Vector3 {
    FP x = 0, y = 0, z = 0;

    Vector3() = default;
    Vector3(FP x_, FP y_, FP z_) : x(x_), y(y_), z(z_) {}

    Vector3 operator+(const Vector3& r) const {
        return {x + r.x, y + r.y, z + r.z};
    }
    Vector3 operator-(const Vector3& r) const {
        return {x - r.x, y - r.y, z - r.z};
    }
    Vector3 operator*(FP s) const {
        return {x * s, y * s, z * s};
    }
    Vector3& operator+=(const Vector3& r) {
        x += r.x; y += r.y; z += r.z;
        return *this;
    }
    Vector3 operator/(FP scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }
};

constexpr FP ELECTRON_CHARGE = -1.602e-19;
constexpr FP ELECTRON_MASS   = 9.109e-31;
constexpr FP LIGHT_VELOCITY  = 2.998e8;

struct Particle {
    Vector3 r;
    Vector3 v;
};
