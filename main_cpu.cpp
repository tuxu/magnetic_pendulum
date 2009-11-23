// Compile with: g++-4.0 main_cpu.cpp -o main_cpu -Wall -O3 -I/opt/local/include/ -L/opt/local/lib -lIL -m64
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
using namespace std;

typedef float floating;

void rhs(const floating t, const floating y[], floating yout[]);

void rk4(floating yout[], floating y[], floating dydt[], const size_t n,
         const floating t, const floating h) {
    floating dym[n], dyt[n], yt[n];
    floating hh = h * 0.5;
    floating h6 = h / 6.0;
    floating th = t + hh;

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

void rkdumb(floating yout[], floating y[], const size_t n, const floating t1,
            const floating t2, const size_t steps) {
    floating vout[n], v[n], dv[n];
    
    for (size_t i = 0; i < n; ++i) {
        v[i] = y[i];
    }

    floating t = t1;
    floating h = (t2 - t1) / steps;
    
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
void rk45_step(const floating y[], const floating dydt[], const size_t n,
               const floating t, const floating h,
               floating yout[], floating dydtout[], floating yerr[]) {
    // Coefficients
    // <http://en.wikipedia.org/wiki/Dormand-Prince>
    const floating c2 = 1./5, c3 = 3./10, c4 = 4./5, c5 = 8./9;
    const floating a21 = 1./5;
    const floating a31 = 3./40, a32 = 9./40;
    const floating a41 = 44./45, a42 = -56./15, a43 = 32./9;
    const floating a51 = 19372./6561, a52 = -25360./2187, a53 = 64448./6561,
                a54 = -212./729;
    const floating a61 = 9017./3168, a62 = -355./33, a63 = 46732./5247,
                a64 = 49./176, a65 = -5103./18656;
    const floating a71 = 35./384, a73 = 500./1113, a74 = 125./192,
                a75 = -2187./6784, a76 = 11./84;
    // b coefficients for the reduced solution (4th order)
    const floating b1r = 5179./57600, b3r = 7571./16695, b4r = 393./640,
                b5r = -92097./339200., b6r = 187./2100, b7r = 1./40;
    // b coefficients for the 5th order solution
    const floating b1 = 35./384, b3 = 500./1113, b4 = 125./192,
                b5 = -2187./6784, b6 = 11./84;

    floating tmp[n];

    // Slope 1
    const floating *k1 = dydt;

    // Slope 2
    floating k2[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + a21 * h * k1[i];
    }
    rhs(t + c2 * h, tmp, k2);

    // Slope 3
    floating k3[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    }
    rhs(t + c3 * h, tmp, k3);

    // Slope 4
    floating k4[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    }
    rhs(t + c4 * h, tmp, k4);

    // Slope 5
    floating k5[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + 
                             a54 * k4[i]);
    }
    rhs(t + c5 * h, tmp, k5);

    // Slope 6
    floating k6[n];
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] +
                             a64 * k4[i] + a65 * k5[i]);
    }
    rhs(t + h, tmp, k6);

    // Slope 7
    // use FSAL trick to avoid one extra function evaluation
    floating *k7 = dydtout;
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = y[i] + h * (a71 * k1[i]               + a73 * k3[i] +
                             a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
    }
    rhs(t + h, tmp, k7);

    // Solutions
    floating yt_r[n];
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
floating rk45_error(const floating y[], const floating yh[], const floating yerr[],
                 const size_t n, const floating atol, const floating rtol) {
    floating err = 0;
    for (size_t i = 0; i < n; ++i) {
        // scale = atol + max(y,yh) * rtol
        floating scale = atol + rtol * fmax(abs(y[i]), abs(yh[i]));
        floating w = (yerr[i] / scale);
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
void rk45(const floating y[], const size_t n, const floating t, const floating dt,
          floating yout[],
          const floating atol, const floating rtol, const size_t mxsteps = 2000,
          floating hmin = -1) {
    const floating safety = 0.9; // Safety factor.
    const floating alpha = 0.2, minscale = 0.2, maxscale = 10.0;

    /* h should not go below this value or the integration will exceed the
     * maximum number of steps.
     */
    if (hmin < 0) {
        hmin = dt / (mxsteps - 1);
    }

    // First try with given dt.
    floating h = dt;
    
    floating cur_t  = t;  
    floating target_t = t + dt;

    // Preparation
    floating dydt[n], dydt_out[n];
    floating yt[n], yt_out[n];
    floating yt_err[n];

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
        floating err = rk45_error(yt, yt_out, yt_err, n, atol, rtol);

        floating hnew, scale;
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
    floating gamma;            // Friction coefficient.
    int exponent;           // Potential exponent.
    int N;                  // Number of magnets.
    floating *alphas;          // Array of magnet strengths.
    floating *rns;             /* Array of the correspondent magnet positions,
                               magnet i's cartesian coordinates being
                               x = 3 * i + 0, y = 3 * i + 1, z = 3 * i + 2.
                             */
} parameters;

/*
 * Calculates the right hand side of the magnetic pendulum's ODE.
 *
 */
void rhs(const floating t, const floating y[], floating yout[]) {
    // Vector
    floating phi = y[0], theta = y[1], phidot = y[2], thetadot = y[3];

    // Get boundary conditions.
    floating gamma = parameters.gamma;
    floating exponent = parameters.exponent;
    int N = parameters.N;
    floating *alphas = parameters.alphas, *rns = parameters.rns;

    // Minimize trigonometric calculations.
    floating cp = cos(phi);
    floating sp = sin(phi);
    floating ct = cos(theta);
    floating st = sin(theta);
    floating tt = tan(theta);

    // Sum the magnet's contributions to the nominator.
    floating sum_theta = 0;
    floating sum_phi = 0;
    for (int i = 0; i < N; ++i) {
        // Magnet coordinates and strength.
        floating x = rns[3*i+0];
        floating y = rns[3*i+1];
        floating z = rns[3*i+2];
        floating alpha = alphas[i];
        // Denominator
        floating A = pow(
                    pow(cp * st - x, (floating)2.0) +
                    pow(sp * st - y, (floating)2.0) +
                    pow(ct - z, (floating)2.0),
                    floating(exponent / 2.0 + 1.0));
        sum_theta += exponent * alpha / A *
            (ct*cp * (cp*st - x) + ct*sp * (st*sp - y) - st * (ct - z));
        sum_phi += exponent * alpha / A *
            (cp * (sp - y/st) - sp * (cp - x/st));
    }

    // Evaluate second derivatives.
    floating thetadotdot = ct*st * phidot*phidot + st - gamma * thetadot - sum_theta;
    floating phidotdot = -gamma * st * phidot - 2.0/tt * thetadot * phidot - sum_phi;

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
floating get_kinetic(const floating *y) {
    return 0.5 * (y[3]*y[3] + sin(y[1])*sin(y[1]) * y[2]*y[2]);
}

/*
 * Returns the potential energy of the system.
 *
 */
floating get_potential(const floating *y) {
    floating cp = cos(y[0]);
    floating sp = sin(y[0]);
    floating ct = cos(y[1]);
    floating st = sin(y[1]);
    
    floating sum = 0;
    for (int i = 0; i < parameters.N; ++i) {
        sum += parameters.alphas[i] * pow(
                pow(cp*st - parameters.rns[3*i+0], floating(2.0)) +
                pow(sp*st - parameters.rns[3*i+1], floating(2.0)) +
                pow(ct - parameters.rns[3*i+2], floating(2.0)),
                floating(parameters.exponent / 2.0)
                );
    }

    return ct - sum;
}

/*
 * Returns the distances of the current position to the magnets.
 *
 */
void distances_to_magnets(const floating *y, floating *distances) {
    floating cp = cos(y[0]);
    floating sp = sin(y[0]);
    floating ct = cos(y[1]);
    floating st = sin(y[1]);
    
    for (int i = 0; i < parameters.N; ++i) {
        floating dx = cp*st - parameters.rns[3*i+0];
        floating dy = sp*st - parameters.rns[3*i+1];
        floating dz = ct - parameters.rns[3*i+2];
        distances[i] = sqrt(dx*dx + dy*dy + dz*dz);
    }
}

/*
 * Returns the magnet the pendulum be next to for large times and given
 * initial position `phi' and `theta'.
 *
 */
int find_magnet(floating phi, floating theta) {
    // Constants
    const floating time_step = 5.0;
    const floating atol = 1e-6, rtol = 1e-6;
    const int max_iterations = 30;
    const floating min_kin = 0.5;

    // Starting vector.
    floating y[] = { phi, theta, 0, 0 };

    // What to do for `theta' = 0 or pi?
    floating eps = 1e-6;
    if (theta > M_PI-eps && theta < M_PI+eps)
        return -1;
    if (theta > -eps && theta < eps)
        return -1;

    int last_magnet = -1;
    floating y_tmp[4];
    floating dist[parameters.N];

    for (int iterations = 0; iterations < max_iterations; ++iterations) {
        // Solve ODE for t + time_step.
        //rkdumb(y_tmp, y, 4, 0, time_step, time_count);
        rk45(y, 4, 0, time_step, y_tmp, atol, rtol);

        for (int i = 0; i < 4; ++i)
            y[i] = y_tmp[i];

        // Find the magnet that is nearest.
        distances_to_magnets(y, dist);
        int magnet = -1;
        floating min = -1;
        for(int i = 0; i < parameters.N; ++i) {
            if (min < 0 || dist[i] < min) {
                min = dist[i];
                magnet = i;
            }
        }
        
        floating kin = get_kinetic(y);

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


#include <IL/il.h>

floating phi_from = 0.0, phi_to = 2*M_PI;
floating theta_from = 0.5 * M_PI, theta_to = M_PI;
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
    floating alphas[3] = { 1.0, 1.0, 1.0 };
    floating rns[9] = { -0.8660254, -0.5, -1.3,
                      0.8660254, -0.5, -1.3,
                      0.0, 1.0, -1.3 };
    parameters.alphas = alphas;
    parameters.rns = rns;
}

void magnet_map() {
    floating dphi = (phi_to - phi_from) / (phi_steps - 1);
    floating dtheta = (theta_to - theta_from) / (theta_steps - 1);
    floating phi = 0, theta = 0;
    
    // Time measurement.
    char buf[100];
    time_t t_start = time(0);
    strftime(buf, 100, "%c", localtime(&t_start));
    cout << "Start: " << buf << endl;
    
    for (int x = 0; x < phi_steps; ++x) {
        for (int y = 0; y < theta_steps; ++y) {
            // Determine current position.
            phi = phi_from + x * dphi;
            theta = theta_from + y * dtheta;

            // Progress indicator.
            floating progress = 100.0 * (x*theta_steps+y+1) / (phi_steps*theta_steps);
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

    time_t t_end = time(0);
    strftime(buf, 100, "%c", localtime(&t_end));
    double t_diff = std::difftime(t_end, t_start);
    cout << "End: " << buf << endl;
    cout << "Mapping took " << t_diff << " s!" << endl;
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

