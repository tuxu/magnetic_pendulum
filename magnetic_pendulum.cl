/*
 * OpenCL kernel for the Magnetic Pendulum 
 *
 */

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// Integration tolerances
const float atol = 1e-6, rtol = 1e-6;

/* 
 * Structure containing experiment parameters.
 *
 */
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

// -----------------------------------------------------------------------------
// Prototypes
// -----------------------------------------------------------------------------
void rk45(const float y[], const size_t n, const float t, const float dt,
          float yout[], const float atol, const float rtol,
          const size_t mxsteps, float hmin);

int find_magnet(float phi, float theta);


// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

__kernel int square(__global float *input, __global float *output,
                    const unsigned int count) {
    int i = get_global_id(0);
}


// -----------------------------------------------------------------------------
// Magnetic Pendulum
// -----------------------------------------------------------------------------

/*
 * Calculates the right hand side of the magnetic pendulum's ODE.
 *
 */
void rhs(const float t, const float y[], float yout[]) {
    // Vector
    float phi = y[0], theta = y[1], phidot = y[2], thetadot = y[3];

    // Get boundary conditions.
    float gamma = parameters.gamma;
    float exponent = parameters.exponent;
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
        // Denominator
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

/*
 * Returns the kinetic energy of the system.
 *
 */
float get_kinetic(const float *y) {
    return 0.5 * (y[3]*y[3] + sin(y[1])*sin(y[1]) * y[2]*y[2]);
}

/*
 * Returns the potential energy of the system.
 *
 */
float get_potential(const float *y) {
    float cp = cos(y[0]);
    float sp = sin(y[0]);
    float ct = cos(y[1]);
    float st = sin(y[1]);
    
    float sum = 0;
    for (int i = 0; i < parameters.N; ++i) {
        sum += parameters.alphas[i] * pow(
                pow(cp*st - parameters.rns[3*i+0], 2.0f) +
                pow(sp*st - parameters.rns[3*i+1], 2.0f) +
                pow(ct - parameters.rns[3*i+2], 2.0f),
                parameters.exponent / 2.0f
                );
    }

    return ct - sum;
}

/*
 * Returns the distances of the current position to the magnets.
 *
 */
void distances_to_magnets(const float *y, float *distances) {
    float cp = cos(y[0]);
    float sp = sin(y[0]);
    float ct = cos(y[1]);
    float st = sin(y[1]);
    
    for (int i = 0; i < parameters.N; ++i) {
        float dx = cp*st - parameters.rns[3*i+0];
        float dy = sp*st - parameters.rns[3*i+1];
        float dz = ct - parameters.rns[3*i+2];
        distances[i] = sqrt(dx*dx + dy*dy + dz*dz);
    }
}

/*
 * Returns the magnet the pendulum be next to for large times and given
 * initial position `phi' and `theta'.
 *
 */
int find_magnet(float phi, float theta) {
    // Constants
    const float time_step = 5.0;
    const int max_iterations = 30;
    const float min_kin = 0.5;

    // Starting vector.
    float y[] = { phi, theta, 0, 0 };

    // What to do for `theta' = 0 or pi?
    float eps = 1e-6;
    if (theta > M_PI-eps && theta < M_PI+eps)
        return -1;
    if (theta > -eps && theta < eps)
        return -1;

    int last_magnet = -1;
    float y_tmp[4];
    float dist[parameters.N];

    for (int iterations = 0; iterations < max_iterations; ++iterations) {
        // Solve ODE for t + time_step.
        rk45(y, 4, 0, time_step, y_tmp, atol, rtol, 10000, -1);

        for (int i = 0; i < 4; ++i)
            y[i] = y_tmp[i];

        // Find the magnet that is nearest.
        distances_to_magnets(y, dist);
        int magnet = -1;
        float min = -1;
        for(int i = 0; i < parameters.N; ++i) {
            if (min < 0 || dist[i] < min) {
                min = dist[i];
                magnet = i;
            }
        }
        
        float kin = get_kinetic(y);

        /* Stop search, if:
         *      - kinetic energy is below threshold
         *      - pendulum didn't move to another magnet within the last
         *        iteration
         */
        if (kin < min_kin && magnet == last_magnet) {
            break;
        } else {
            last_magnet = magnet;
        }
    }

    return last_magnet;
}


// -----------------------------------------------------------------------------
// ODE integration
// -----------------------------------------------------------------------------

/*
 * Do an integration step using 5th order Dormand-Prince Runge-Kutta method.
 *
 * Parameters:
 *       y -- y(t)
 *    dydt -- y'(t, y)
 *       n -- number of vector elements
 *       t -- t
 *       h -- step width
 * Output:
 *    yout -- y(t + h)
 * dydtout -- y'(t + h, yout)
 *    yerr -- difference of 5th and 4th order solution
 *
 */
void rk45_step(const float y[], const float dydt[], const size_t n,
               const float t, const float h,
               float yout[], float dydtout[], float yerr[]) {
    // Coefficients
    // <http://en.wikipedia.org/wiki/Dormand-Prince>
    const float c2 = 1./5, c3 = 3./10, c4 = 4./5, c5 = 8./9;
    const float a21 = 1./5;
    const float a31 = 3./40, a32 = 9./40;
    const float a41 = 44./45, a42 = -56./15, a43 = 32./9;
    const float a51 = 19372./6561, a52 = -25360./2187, a53 = 64448./6561,
                a54 = -212./729;
    const float a61 = 9017./3168, a62 = -355./33, a63 = 46732./5247,
                a64 = 49./176, a65 = -5103./18656;
    const float a71 = 35./384, a73 = 500./1113, a74 = 125./192,
                a75 = -2187./6784, a76 = 11./84;
    // b coefficients for the reduced solution (4th order)
    const float b1r = 5179./57600, b3r = 7571./16695, b4r = 393./640,
                b5r = -92097./339200., b6r = 187./2100, b7r = 1./40;
    // b coefficients for the 5th order solution
    const float b1 = 35./384, b3 = 500./1113, b4 = 125./192,
                b5 = -2187./6784, b6 = 11./84;

    float tmp[n];

    // Slope 1
    const float *k1 = dydt;

    // Slope 2
    float k2[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + a21 * h * k1[i];
    }
    rhs(t + c2 * h, tmp, k2);

    // Slope 3
    float k3[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    }
    rhs(t + c3 * h, tmp, k3);

    // Slope 4
    float k4[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    }
    rhs(t + c4 * h, tmp, k4);

    // Slope 5
    float k5[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + 
                             a54 * k4[i]);
    }
    rhs(t + c5 * h, tmp, k5);

    // Slope 6
    float k6[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] +
                             a64 * k4[i] + a65 * k5[i]);
    }
    rhs(t + h, tmp, k6);

    // Slope 7
    // use FSAL trick to avoid one extra function evaluation
    float *k7 = dydtout;
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a71 * k1[i]               + a73 * k3[i] +
                             a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
    }
    rhs(t + h, tmp, k7);

    // Solutions
    float yt_r[n];
    for (size_t i = 0; i < n; ++i) {
        // 4th order solution
        yt_r[i] = y[i] + h * (b1r * k1[i]               + b3r * k3[i] +
                              b4r * k4[i] + b5r * k5[i] + b6r * k6[i] +
                              b7r * k7[i]);
        // 5th order solution
        yout[i] = y[i] + h * (b1 * k1[i]              + b3 * k3[i] +
                              b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
        // Error estimation
        yerr[i] = yout[i] - yt_r[i];
    }

}

/*
 * Calculates and returns the error of a given step.
 *
 */
float rk45_error(const float y[], const float yh[], const float yerr[],
                 const size_t n, const float atol, const float rtol) {
    float err = 0;
    for (size_t i = 0; i < n; ++i) {
        // scale = atol + max(y,yh) * rtol
        float scale = atol + rtol * fmax(fabs(y[i]), fabs(yh[i]));
        float w = (yerr[i] / scale);
        err += w * w;
    }

    return sqrt(err / n);
}

/* 
 * Adaptive step width driver for rk45_step.
 *
 * Parameters:
 *       y -- starting vector
 *       n -- number of components
 *       t -- independent parameter
 *      dt -- step to reach
 *    atol -- absolute tolerance
 *    rtol -- relative tolerance
 * mxsteps -- maximum number of steps to do
 *    hmin -- minimum helper step width, hmin < 0 for automatic choice
 *
 * Output:
 *    yout -- Output vector
 *
 */
void rk45(const float y[], const size_t n, const float t, const float dt,
          float yout[],
          const float atol, const float rtol, const size_t mxsteps,
          float hmin) {
    const float safety = 0.9; // Safety factor.
    const float alpha = 0.2, minscale = 0.2, maxscale = 10.0;

    /* h should not go below this value or the integration will exceed the
     * maximum number of steps.
     */
    if (hmin < 0) {
        hmin = dt / (mxsteps - 1);
    }

    // First try with given dt.
    float h = dt;
    
    float cur_t  = t;  
    float target_t = t + dt;

    // Preparation
    float dydt[n], dydt_out[n];
    float yt[n], yt_out[n];
    float yt_err[n];

    rhs(t, y, dydt);
    for (size_t i = 0; i < n; ++i) {
        yt[i] = y[i];        
    }

    bool last_rejected = false;
    bool last_adjusted = true;

    // Walk through (t..t+dt) and adapt step width.
    for (size_t steps = 0; ; ++steps) {
        // Maximum steps used?
        if (steps > mxsteps) {
            /*printf("Used maximum number of steps!\n"
                   "Remaining: %f\n"
                   "h: %f\n", (target_t - cur_t), h);*/
            break;
        }

        // Will exceed integration interval?
        if (cur_t + h > target_t) {
            break; 
        }

        // Do an actual integration step.
        rk45_step(yt, dydt, n, cur_t, h, yt_out, dydt_out, yt_err);
        float err = rk45_error(yt, yt_out, yt_err, n, atol, rtol);

        float hnew, scale;
        if (err <= 1 || !last_adjusted) {
            // Integration was successful. Compute a better h.
            if (err == 0) {
                scale = maxscale;
            } else {
                scale = safety * pow(err, -alpha);
                scale = fmin(fmax(scale, minscale), maxscale);
            }

            if (last_rejected)
                hnew = h * fmin(scale, 1.0f);
            else
                hnew = h * scale;

            last_rejected = false;
            
            cur_t += h;
            for (size_t i = 0; i < n; ++i) {
                yt[i] = yt_out[i];
                dydt[i] = dydt_out[i];
            }
        } else {
            // Error too big.
            scale = fmax(safety * pow(err, -alpha), minscale);
            hnew = h * scale;
            last_rejected = true;
        }

        if (hnew < hmin) {
            //printf("h got too small, no adjustment.\n");
            last_adjusted = false;
        } else {
            h = hnew;
            last_adjusted = true;
        }
    }

    // Do the remaining step to reach target_t.
    rk45_step(yt, dydt, n, cur_t, target_t - cur_t, yout, dydt_out, yt_err);
}



