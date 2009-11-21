/*
 * OpenCL kernel for the Magnetic Pendulum 
 * (c) 2009, Tino Wagner
 *
 */

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288f
#endif

#define SIN(x) native_sin(x)
#define COS(x) native_cos(x)
#define TAN(x) native_tan(x)
#define SQRT(x) native_sqrt(x)

// Integration tolerances
#define ATOL 1e-6f
#define RTOL ATOL

// Parameters
float friction;
int exponent;
int n_magnets;
__global float *alphas;
__global float *rns;

// -----------------------------------------------------------------------------
// Prototypes
// -----------------------------------------------------------------------------
void rk45(const float4 *y, const float t, const float dt, float4 *yout,
          const float atol, const float rtol, const size_t mxsteps, float hmin);

int find_magnet(const float phi, const float theta,
                const float time_step,
                const float min_kin,
                const unsigned int max_iterations);


// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

__kernel void map_magnets(__global float *coords,
                         __global int *magnets,
                         const unsigned int offset,
                         const unsigned int count,
                         const float friction_,
                         const int exponent_,
                         const unsigned int n_magnets_,
                         __global float *alphas_,
                         __global float *rns_,
                         const float time_step,
                         const float min_kin,
                         const unsigned int max_iterations) {
    size_t i = get_global_id(0);
    
    if (i < count) {
        size_t ind = offset + i;
        
        friction = friction_;
        exponent = exponent_;
        n_magnets = n_magnets_;
        alphas = alphas_;
        rns = rns_;
        
        // Fetch coordinates.
        float phi = coords[ 2 * ind + 0];
        float theta = coords[2 * ind + 1];
        
        int magnet = find_magnet(phi, theta, time_step, min_kin,
                                 max_iterations);
        magnets[ind] = magnet;
    }
}


// -----------------------------------------------------------------------------
// Magnetic Pendulum
// -----------------------------------------------------------------------------

/*
 * Calculates the right hand side of the magnetic pendulum's ODE.
 *
 */
void rhs(const float t, const float4 *y, float4 *dydt) {
    // Minimize trigonometric calculations.
    const float cp = COS((*y).s0);
    const float sp = SIN((*y).s0);
    const float ct = COS((*y).s1);
    const float st = SIN((*y).s1);
    const float tt = TAN((*y).s1);

    // Sum the magnet's contributions to the nominator.
    float sum_theta = 0;
    float sum_phi = 0;
    for (int i = 0; i < n_magnets; ++i) {
        // Magnet coordinates and strength.
        float x = rns[3 * i + 0];
        float y = rns[3 * i + 1];
        float z = rns[3 * i + 2];
        float alpha = alphas[i];
        // Denominator
        float A = pow(
                    pown(cp * st - x, 2) +
                    pown(sp * st - y, 2) +
                    pown(ct - z, 2),
                    - exponent / 2.0f - 1.0f);
        sum_theta += exponent * alpha * A *
            (ct*cp * (cp*st - x) + ct*sp * (st*sp - y) - st * (ct - z));
        sum_phi += exponent * alpha * A *
            (cp * (sp - y/st) - sp * (cp - x/st));
    }

    // Evaluate second derivatives.
    const float thetadotdot = ct*st * (*y).s2 * (*y).s2 + st
                            - friction * (*y).s3 - sum_theta;
    const float phidotdot = -friction * st * (*y).s2 
                            - 2.0f/tt * (*y).s3 * (*y).s2 - sum_phi;

    // Return vector.
    *dydt = (float4)((*y).s2, (*y).s3, phidotdot, thetadotdot);
}

/*
 * Returns the kinetic energy of the system.
 *
 */
float get_kinetic(const float4 *y) {
    return 0.5f * ((*y).s3*(*y).s3 + SIN((*y).s1)*SIN((*y).s1) * (*y).s2*(*y).s2);
}

/*
 * Returns the potential energy of the system.
 *
 */
float get_potential(const float4 *y) {
    const float cp = COS((*y).s0);
    const float sp = SIN((*y).s0);
    const float ct = COS((*y).s1);
    const float st = SIN((*y).s1);
    
    float sum = 0;
    for (int i = 0; i < n_magnets; ++i) {
        sum += alphas[i] * pow(
                pown(cp * st - rns[3*i+0], 2) +
                pown(sp * st - rns[3*i+1], 2) +
                pown(ct - rns[3*i+2], 2),
                exponent / 2.0f
                );
    }

    return ct - sum;
}

/*
 * Returns the magnet the pendulum be next to for large times and given
 * initial position `phi' and `theta'.
 *
 */
int find_magnet(const float phi, const float theta,
                const float time_step,
                const float min_kin,
                const unsigned int max_iterations) {
    // Starting vector.
    float4 y = (float4)(phi, theta, 0, 0);

    // What to do for `theta' = 0 or pi?
    float eps = 1e-6f;
    if (theta > M_PI-eps && theta < M_PI+eps)
        return 0;
    if (theta > -eps && theta < eps)
        return 0;

    int last_magnet = -1;

    for (int iterations = 0; iterations < max_iterations; ++iterations) {
        // Solve ODE for t + time_step.
        rk45(&y, 0, time_step, &y, ATOL, RTOL, 10000, -1);

        // Find the magnet that is nearest.
        int magnet = -1;
        float min = -1;
        
        const float4 r_pendulum = (float4)(COS(y.s0) * SIN(y.s1),
                                           SIN(y.s0) * SIN(y.s1),
                                           COS(y.s1), 0);
        for (int i = 0; i < n_magnets; ++i) {
            // Calculate distance to magnet i.
            float4 r_magnet = (float4)(rns[3*i+0], rns[3*i+1],
                                       rns[3*i+2], 0);
            float dist = distance(r_pendulum, r_magnet);
            
            if (min < 0 || dist < min) {
                min = dist;
                magnet = i;
            }
        }
        
        float kin = get_kinetic(&y);

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
 *       t -- t
 *       h -- step width
 * Output:
 *    yout -- y(t + h)
 * dydtout -- y'(t + h, yout)
 *    yerr -- difference of 5th and 4th order solution
 *
 */
void rk45_step(const float4 *y, const float4 *dydt,
               const float t, const float h,
               float4 *yout, float4 *dydtout, float4 *yerr) {
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

    float4 tmp, k2, k3, k4, k5, k6;

    // Slope 1: k1 = dydt

    // Slope 2
    tmp = *y + a21 * h * *dydt;
    rhs(t + c2 * h, &tmp, &k2);

    // Slope 3
    tmp = *y + h * (a31 * *dydt + a32 * k2);
    rhs(t + c3 * h, &tmp, &k3);

    // Slope 4
    tmp = *y + h * (a41 * *dydt + a42 * k2 + a43 * k3);
    rhs(t + c4 * h, &tmp, &k4);

    // Slope 5
    tmp = *y + h * (a51 * *dydt + a52 * k2 + a53 * k3 + a54 * k4);
    rhs(t + c5 * h, &tmp, &k5);

    // Slope 6
    tmp = *y + h * (a61 * *dydt + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5);
    rhs(t + h, &tmp, &k6);

    // Slope 7
    // use FSAL trick to avoid one extra function evaluation
    tmp = *y + h * (a71 * *dydt + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6);
    rhs(t + h, &tmp, dydtout);

    // Solutions
    // 4th order solution
    float4 yt_r = *y + h * (b1r * *dydt + b3r * k3 + b4r * k4 + b5r * k5 +
                            b6r * k6 + b7r * *dydtout);
    // 5th order solution
    *yout = *y + h * (b1 * *dydt + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
    
    // Error estimation
    *yerr = *yout - yt_r;

}

/*
 * Calculates and returns the error of a given step.
 *
 */
float rk45_error(const float4 *y, const float4 *yh, const float4 *yerr,
                 const float atol, const float rtol) {
    // scale = atol + max(y,yh) * rtol
    float4 scale = (float4)atol + (float4)rtol * fmax(fabs(*y), fabs(*yh));
    float4 w = *yerr / scale;

    return length(w) / 2.0f;
}

/* 
 * Adaptive step width driver for rk45_step.
 *
 * Parameters:
 *       y -- starting vector
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
void rk45(const float4 *y, const float t, const float dt, float4 *yout,
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
    float4 dydt, dydt_out;
    float4 yt_out;
    float4 yt_err;

    rhs(t, y, &dydt);
    *yout = *y;

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
        rk45_step(yout, &dydt, cur_t, h, &yt_out, &dydt_out, &yt_err);
        float err = rk45_error(yout, &yt_out, &yt_err, atol, rtol);

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
            
            *yout = yt_out;
            dydt = dydt_out;
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
    rk45_step(yout, &dydt, cur_t, target_t - cur_t, yout, &dydt_out, &yt_err);
}



