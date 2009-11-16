#include <iostream>
#include <cmath>
using namespace std;

void rk4(float yout[], float y[], float dydt[], const size_t n,
         const float t, const float h,
         void derivs(const float t, const float y[], float yout[])) {
    float dym[n], dyt[n], yt[n];
    float hh = h * 0.5;
    float h6 = h / 6.0;
    float th = t + hh;

    // First step.
    for (size_t i = 0; i < n; ++i)
        yt[i] = y[i] + hh * dydt[i];

    // Second step.
    derivs(th, yt, dyt);
    for (size_t i = 0; i < n; ++i)
        yt[i] = y[i] + hh * dyt[i];

    // Third step.
    derivs(th, yt, dym);
    for (size_t i = 0; i < n; ++i) {
        yt[i] = y[i] + h * dym[i];
        dym[i] += dyt[i];
    }

    // Fourth step.
    derivs(t + h, yt, dyt);

    // Produce output.
    for (size_t i = 0; i < n; ++i) {
        yout[i] = y[i] + h6 * (dydt[i] + dyt[i] + 2.0 * dym[i]);
    }

}

struct Parameters {
    float gamma;            // Friction coefficient.
    int exponent;           // Potential exponent.
    int N;                  // Number of magnets.
    float *alphas;          // Array of magnet strengths.
    float *rns;             /* Array of the correspondent magnet positions,
                               magnet i's cartesian coordinates being
                               x = 3 * i + 0, y = 3 * i + 1, z = 3 * i + 2.
                             */
} parameters;

void derivs(const float t, const float y[], float yout[]) {
    // Vector
    float phi = y[0], theta = y[1], phidot = y[2], thetadot = y[3];

    // Get boundary conditions.
    float gamma = parameters.gamma;
    int exponent = parameters.exponent;
    int N = parameters.N;
    float *alphas = parameters.alphas, *rns = parameters.rns;

    // Minimize trigonometric calculations.
    float cp = cos(phi);
    float sp = sin(phi);
    float ct = cos(theta);
    float st = sin(theta);
    float tt = tan(theta);

    // Sum the magnet's contributions to the nominator.
    float sum_theta = 0;
    float sum_phi = 0;
    for (int i = 0; i < N; ++i) {
        // Magnet coordinates and strength.
        float x = rns[3*i+0];
        float y = rns[3*i+1];
        float z = rns[3*i+2];
        float alpha = alphas[i];
        // Nominator
        float A = pow(
                    pow(cp * st - x, 2.0f) +
                    pow(sp * st - y, 2.0f) +
                    pow(ct - z, 2.0f),
                    exponent / 2.0f + 1.0f);
        sum_theta += exponent * alpha / A *
            (ct*cp * (cp*st - x) + ct*sp * (st*sp - y) - st * (ct - z));
        sum_phi += exponent * alpha / A *
            (cp * (sp - y/st) - sp * (cp - x/st));
    }

    // Evaluate second derivatives.
    float thetadotdot = ct*st * phidot*phidot + st - gamma * thetadot - sum_theta;
    float phidotdot = -gamma * st * phidot - 2.0/tt * thetadot * phidot - sum_phi;

    // Return vector.
    yout[0] = phidot;
    yout[1] = thetadot;
    yout[2] = phidotdot;
    yout[3] = thetadotdot;
}

int main (int argc, const char *argv[]) {
    // Experiment setup.
    parameters.gamma = 0.1;
    parameters.exponent = 2;
    parameters.N = 3;
    float alphas[3] = { 1.0, 1.0, 1.0 };
    float rns[9] = { -0.8660254, -0.5, -1.3,
                      0.8660254, -0.5, -1.3,
                      0.0, 1.0, -1.3 };
    parameters.alphas = alphas;
    parameters.rns = rns;
}

