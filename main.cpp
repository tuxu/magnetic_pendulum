#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

void rhs(const float t, const float y[], float yout[]);

void rk4(float yout[], float y[], float dydt[], const size_t n,
         const float t, const float h) {
    float dym[n], dyt[n], yt[n];
    float hh = h * 0.5;
    float h6 = h / 6.0;
    float th = t + hh;

    // First step.
    for (size_t i = 0; i < n; ++i)
        yt[i] = y[i] + hh * dydt[i];

    // Second step.
    rhs(th, yt, dyt);
    for (size_t i = 0; i < n; ++i)
        yt[i] = y[i] + hh * dyt[i];

    // Third step.
    rhs(th, yt, dym);
    for (size_t i = 0; i < n; ++i) {
        yt[i] = y[i] + h * dym[i];
        dym[i] += dyt[i];
    }

    // Fourth step.
    rhs(t + h, yt, dyt);

    // Produce output.
    for (size_t i = 0; i < n; ++i) {
        yout[i] = y[i] + h6 * (dydt[i] + dyt[i] + 2.0 * dym[i]);
    }

}

void rkdumb(float yout[], float y[], const size_t n, const float t1,
            const float t2, const size_t steps) {
    float vout[n], v[n], dv[n];
    
    for (size_t i = 0; i < n; ++i) {
        v[i] = y[i];
    }

    float t = t1;
    float h = (t2 - t1) / steps;
    
    for (size_t k = 0; k < steps; ++k) {
        rhs(t, v, dv);
        rk4(vout, v, dv, n, t, h);
        //assert ((t+h) != t);
        t += h;
        for (size_t i = 0; i < n; ++i) {
            v[i] = vout[i];
        }
    }

    for (size_t i = 0; i < n; ++i) {
        yout[i] = v[i];
    }
    
}

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
        float scale = atol + rtol * fmax(abs(y[i]), abs(yh[i]));
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
 *    hmin -- minimum helper step width
 *
 * Output:
 *    yout -- Output vector
 *
 */
void rk45(const float y[], const size_t n, const float t, const float dt,
          float yout[],
          const float atol, const float rtol, const size_t mxsteps = 10000,
          float hmin = -1) {
    const float safety = 0.9; // Safety factor.
    const float alpha = 0.2, minscale = 0.2, maxscale = 10.0;

    /* h should not go below this value or the integration will exceed the
     * maximum number of steps.
     */
    if (hmin < 0) {
        hmin = dt / mxsteps;
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
            cerr << "Used maximum number of steps!" << endl;
            cerr << "Remaining: " << (target_t - cur_t) << endl;
            cerr << "h: " << h << endl;
            //exit(1);
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
                hnew = h * fmin(scale, 1.0);
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
            //cerr << "h got too small, no adjustment." << endl;
            last_adjusted = false;
        } else {
            h = hnew;
            last_adjusted = true;
        }
    }

    // Do the remaining step to reach target_t.
    rk45_step(yt, dydt, n, cur_t, target_t - cur_t, yout, dydt_out, yt_err);
}

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
    const float atol = 1e-6, rtol = 1e-6;
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
        //rkdumb(y_tmp, y, 4, 0, time_step, time_count);
        rk45(y, 4, 0, time_step, y_tmp, atol, rtol);

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


// --------------------------------------------------------------------------
// Compile with: g++-4.0 main.cpp -o main -Wall -O3 -I/opt/local/include/ -L/opt/local/lib -lIL -m64

#include <IL/il.h>

float phi_from = 0.0, phi_to = 2*M_PI;
float theta_from = 0.5 * M_PI, theta_to = M_PI;
const int phi_steps = 800, theta_steps = 800;
int magnets[phi_steps][theta_steps];
ILubyte pixels[phi_steps * theta_steps * 3];
ILubyte colors[3*3] = {255, 0, 0,
                       0, 255, 0,
                       0, 0, 255 };

void setup() {
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

void magnet_map() {
    float dphi = (phi_to - phi_from) / (phi_steps - 1);
    float dtheta = (theta_to - theta_from) / (theta_steps - 1);
    float phi = 0, theta = 0;
    
    for (int x = 0; x < phi_steps; ++x) {
        for (int y = 0; y < theta_steps; ++y) {
            // Determine current position.
            phi = phi_from + x * dphi;
            theta = theta_from + y * dtheta;

            // Progress indicator.
            float progress = 100.0 * (x*theta_steps+y+1) / (phi_steps*theta_steps);
            cout << showpoint << fixed;
            cout << "\rProgresss: " << setw(6) << setprecision(2) << progress << " %, ";
            cout << "[" << setw(4) << x << ", " << setw(4) << y << "] = ";
            cout << setprecision(4) << "(" << phi << ", " << theta << ")";
            cout.flush();

            // Perform search.
            int magnet = find_magnet(phi, theta);
            magnets[x][y] = magnet;
            if (magnet >= 0) {
                int index = 3 * (phi_steps * (theta_steps - y - 1) + x);
                pixels[index + 0] = colors[3*magnet + 0];
                pixels[index + 1] = colors[3*magnet + 1];  
                pixels[index + 2] = colors[3*magnet + 2];  
            }

            cout << " = " << magnet << "     ";
        }
    }

    cout << endl;
}

void save_image(const char *filename) {
    if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION) {
        printf("DevIL version is different...exiting!\n");
    }    
    ilInit();
    ILuint image;
    ilGenImages(1, &image);
    ilBindImage(image);
    ilTexImage(phi_steps, theta_steps, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, pixels);
    ilEnable(IL_FILE_OVERWRITE);
    ilSaveImage(filename);
    ilDeleteImages(1, &image);
    while (ILenum Error = ilGetError()) {
        printf("Error: 0x%X\n", (unsigned int)Error);
    }
}

void output_magnets() {
    for (int y = 0; y < theta_steps; ++y) {
        for (int x = 0; x < phi_steps; ++x) {
            char c = ' ';
            switch (magnets[x][y]) {
                case 0: c = 'o'; break;
                case 1: c = '#'; break;
                case 2: c = '+'; break;
            }
            cout << c;
        }
        cout << endl;
    }
    cout << endl;
}

int main (int argc, const char *argv[]) {
    setup();
    magnet_map();    
    save_image("test.tif");
    //output_magnets();
}

