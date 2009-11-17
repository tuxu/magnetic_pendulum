#include <iostream>
#include <iomanip>
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
void derivs(const float t, const float y[], float yout[]) {
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
    const int time_count = 5000;
    const float time_dt = time_step / time_count;
    const int max_iterations = 30;
    const float min_kin = 0.5;

    // Starting vector.
    float y[] = { phi, theta, 0, 0 };

    // What to do for `theta' = 0 or pi?
    float eps = 1e-9;
    if (theta > M_PI-eps && theta < M_PI+eps)
        return -1;
    if (theta > -eps && theta < eps)
        return -1;

    int last_magnet = -1;
    float y_tmp[4], dydt[4];
    float dist[parameters.N];

    for (int iterations = 0; iterations < max_iterations; ++iterations) {
        // Solve ODE for t + time_step.
        for (float t = 0; t < time_step; t += time_dt) {
            derivs(t, y, dydt);
            rk4(y_tmp, y, dydt, 4, t, time_dt, derivs);
            for (int i = 0; i < 4; ++i)
                y[i] = y_tmp[i];
        }

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
const int phi_steps = 400, theta_steps = 400;
int magnets[phi_steps][theta_steps];
ILubyte pixels[phi_steps * theta_steps * 3];
ILubyte colors[3*3] = {255, 0, 0,
                       0, 255, 0,
                       0, 0, 255 };

void setup() {
    parameters.gamma = 0.6;
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
                pixels[3*phi_steps*y + 3*x + 0] = colors[3*magnet + 0];
                pixels[3*phi_steps*y + 3*x + 1] = colors[3*magnet + 1];  
                pixels[3*phi_steps*y + 3*x + 2] = colors[3*magnet + 2];  
                /*pixels[3*phi_steps*y + 3*x + 0] = 255.0f*x/(float)phi_steps;
                pixels[3*phi_steps*y + 3*x + 1] = 255.0f*y/(float)theta_steps;
                pixels[3*phi_steps*y + 3*x + 2] = 255;*/
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

