/*
 * Magnetic Pendulum for OpenCL
 * (c) 2009, Tino Wagner
 *
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cstdarg>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <IL/il.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

#undef USE_LLVM_VERSION

#ifdef USE_LLVM_VERSION
const char *kernel_filename = "magnetic_pendulum_llvm.cl";
#else
const char *kernel_filename = "magnetic_pendulum.cl";
#endif

const bool use_gpu = true;

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------

/*
 * OpenCL
 *
 */
int err;                // Error code
cl_device_id device_id; // Compute device ID
cl_context context;     // Compute context
cl_command_queue queue; // Command queue
cl_program program;     // Compute program
cl_kernel kernel;       // Compute kernel

/*
 * Magnetic Pendulum
 *
 */
const float phi_from = 0.0, phi_to = 2 * M_PI;
const float theta_from = 0.5 * M_PI, theta_to = M_PI;
const int phi_steps = 128, theta_steps = 128;
unsigned char colors[3 * 3] = {255, 0, 0,
                               0, 255, 0,
                               0, 0, 255 };
const float friction = 0.1;
const int exponent = 2;
const unsigned int n_magnets = 3;
float alphas[n_magnets] = { 1.0, 1.0, 1.0 };
float rns[3 * n_magnets] = { -0.8660254, -0.5, -1.3,
                              0.8660254, -0.5, -1.3,
                              0.0, 1.0, -1.3 };
const float time_step = 5.0f;
const float min_kin = 0.5f;
const int max_iterations = 30;


// -----------------------------------------------------------------------------
// Functions
// -----------------------------------------------------------------------------

/*
 * Output an error message and quit.
 *
 */
void error(int code, const char *format, ...) {
    char buf[256];
    va_list args;

    va_start(args, format);
    std::vsprintf(buf, format, args);
    std::cerr << "!! Error " << code << ": " << buf << std::endl;
    va_end(args);

    std::exit(code);
}

/*
 * Log a message.
 *
 */
void log(const char *format, ...) {
    char buf[512];
    va_list args;
    
    va_start(args, format);
    std::vsprintf(buf, format, args);
    std::cout << "-- " << buf << std::endl;
    va_end(args);
}

/*
 * Return the file content as a string. 
 *
 */
std::string file_content(const char *filename) {
    std::ifstream in(filename);
    if (!in)
        error(100, "Unable to load file '%s'!", filename);

    std::ostringstream content;

    for (std::string line; !in.eof(); ) {
        std::getline(in, line);
        content << line << std::endl;
    }
    
    return content.str();
}

/*
 * Initializes the compute device and loads the kernel.
 *
 */
void cl_init() {
    log("Connecting to a compute device ...");
    err = clGetDeviceIDs(0, use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU,
                         1, &device_id, 0);
    if (err != CL_SUCCESS)
        error(200, "Failed to create a device group!");

    log("Creating a compute context ...");
    context = clCreateContext(0, 1, &device_id, 0, 0, &err);
    if (!context)
        error(201, "Failed to create a compute context!");

    log("Creating the command queue ...");
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue)
        error(202, "Failed to create a command queue!");

    log("Creating the compute program ...");
    std::string source = file_content(kernel_filename);
    const char *str = source.c_str();

    program = clCreateProgramWithSource(context, 1,
                                        (const char **) &str,
                                        0, &err);
    if (!program)
        error(203, "Failed to create compute program!");

    log("Building the program executable ...");
    err = clBuildProgram(program, 0, 0,
                         "-Werror -cl-mad-enable -cl-fast-relaxed-math", 0, 0);
    if (err != CL_SUCCESS) {
        char *buffer;
        
        for(size_t len = 256; ;) {
            buffer = new char[len];
            size_t sz;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  len, buffer, &sz);
            if (sz <= len) {
                break;
            }
            else {
                delete[] buffer;
                len = sz;
            }
        }
        std::cout << "Program build info:" << std::endl
                  << "--- 8< ---" << std::endl
                  << buffer << std::endl
                  << "--- 8< ---" << std::endl;
        std::cout << "OpenCL error code: " << err << std::endl;
        delete[] buffer;
        
        error(204, "Failed to build program executable!");
    }

    // Find out program binary size
    size_t program_size;
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                           sizeof(program_size), &program_size, 0);
    if (err == CL_SUCCESS) {
        log("Binary size: %d bytes", program_size);
    }

    log("Creating the compute kernel ...");
    
    kernel = clCreateKernel(program, "map_magnets", &err);
    if (!kernel || err != CL_SUCCESS)
        error(205, "Failed to create compute kernel");
}

/*
 * Free OpenCL resources that were previously allocated. 
 *
 */
void cl_deinit() {
    log("Freeing resources ...");
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

/*
 * Output an ASCII art map of the magnets.
 *
 */
void output_magnets(const int *magnets) {
    for (int y = 0; y < theta_steps; ++y) {
        for (int x = 0; x < phi_steps; ++x) {
            char c = ' ';
            switch (magnets[y * phi_steps + x]) {
                case 0: c = 'o'; break;
                case 1: c = '#'; break;
                case 2: c = '+'; break;
            }
            std::cout << c;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/*
 * Save the map as image.
 *
 */
void save_image(const int *magnets, const char *filename) {
    log("Saving magnet map to %s.", filename);
    
    // Create image.
    ILubyte *pixels = new ILubyte[phi_steps * theta_steps * 3];
    float r, g, b;
    for (int y = 0; y < theta_steps; ++y) {
        for (int x = 0; x < phi_steps; ++x) {
            int magnet = magnets[y * phi_steps + x];
            if (magnet < 0) {
                r = g = b = 0;
            } else {
                r = colors[3 * magnet + 0];
                g = colors[3 * magnet + 1];
                b = colors[3 * magnet + 2];
            }
            int index = 3 * (phi_steps * (theta_steps - y - 1) + x);
            pixels[index + 0] = r;
            pixels[index + 1] = g;  
            pixels[index + 2] = b;
        }
    }
    
    // Save to file.
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
    
    delete [] pixels;
}

/*
 * Create a map.
 *
 */
void magnet_map() {
    // Create an array that maps a pixel to coordinates.
    log("Creating the coordinate mapping array ...");
    size_t coords_len = 2 * phi_steps * theta_steps;
    float *coords = new float[coords_len];
    
    float dphi = (phi_to - phi_from) / (phi_steps - 1);
    float dtheta = (theta_to - theta_from) / (theta_steps - 1);
    
    for (int i = 0; i < phi_steps; ++i) {
        for (int j = 0; j < theta_steps; ++j) {
            // Determine current position.
            float phi = phi_from + i * dphi;
            float theta = theta_from + j * dtheta;
            
            // Map the current pixel to the position.
            int index = 2 * (phi_steps * j + i);
            coords[index + 0] = phi;
            coords[index + 1] = theta;
        }
    }
    
    unsigned int compute_units;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                          sizeof(unsigned int), &compute_units, 0);
    if (err != CL_SUCCESS) {
        error(304, "Failed to retrieve compute unit information: %d", err);
    }
    log("Maximum compute units: %d", compute_units);
    
    size_t local;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(size_t), &local, 0);
    if (err != CL_SUCCESS) {
        error(304, "Failed to retrieve kernel work group info: %d", err);
    }
    log("Maximum work group size: %d", local);
    
    size_t cycle_len = compute_units * local;
    size_t n_cycles = (phi_steps * theta_steps) / cycle_len + 1;
    size_t magnets_cl_len = n_cycles * cycle_len;
    log("Using %d cycles with a length of %d.", n_cycles, cycle_len);
    
    log("Asking for device memory ...");
    cl_mem coords_cl = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(float) * coords_len, 0, 0);
    cl_mem alphas_cl = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(float) * n_magnets, 0, 0);
    cl_mem rns_cl = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * 3 * n_magnets, 0, 0);
    cl_mem magnets_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                       sizeof(int) * magnets_cl_len, 0, 0);
    if (!magnets_cl || !coords_cl || !alphas_cl || !rns_cl) {
        error(301, "Could not allocate device memory!");
    }
    
    log("Populating device with global parameters ...");
    err  = clEnqueueWriteBuffer(queue, coords_cl, CL_TRUE, 0,
                                sizeof(float) * coords_len, coords,
                                0, 0, 0);
    err |= clEnqueueWriteBuffer(queue, alphas_cl, CL_TRUE, 0,
                                sizeof(float) * n_magnets, alphas,
                                0, 0, 0);
    err |= clEnqueueWriteBuffer(queue, rns_cl, CL_TRUE, 0,
                                sizeof(float) * 3 * n_magnets, rns,
                                0, 0, 0);
    if (err != CL_SUCCESS) {
        error(302, "Could not write to device memory.");
    }
    
    //log("Setting kernel arguments ...");
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &coords_cl);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &magnets_cl);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &friction);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &exponent);
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &n_magnets);
    err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &alphas_cl);
    err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &rns_cl);
    err |= clSetKernelArg(kernel, 9, sizeof(float), &time_step);
    err |= clSetKernelArg(kernel, 10, sizeof(float), &min_kin);
    err |= clSetKernelArg(kernel, 11, sizeof(unsigned int), &max_iterations);
    if (err != CL_SUCCESS) {
        error(303, "Could no set kernel parameters: %d", err);
    }
    
    // Time measurement.
    char buf[100];
    std::time_t t_start = time(0);
    strftime(buf, 100, "%c", std::localtime(&t_start));
    log("%s: Doing hard work ...", buf);
    
    std::cout << "\r## Progress: " << std::setw(6) << std::setprecision(2) 
              << 0.0f << " % " << std::flush;
    std::cout << std::showpoint << std::fixed;
    // Walk through the array using the pre-calculated cycles and show a
    // progress indicator.
    for (size_t c = 0; c < (n_cycles - 1); ++c) {
        std::cout.flush();
        
        size_t offset = c * cycle_len;
        err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &offset);
        err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &cycle_len);
        if (err != CL_SUCCESS) {
            error(303, "Could no set kernel parameters: %d", err);
        }                    
        
        err = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &cycle_len, 0,
                                     0, 0, 0);
        if (err) {
            error(305, "Failed to execute kernel: %d", err);
        }
        
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            error(306, "Error during kernel execution: %d", err);
        }
        
        // Progress indicator.
        float progress = 100.0 * (c + 1) / (n_cycles - 1);
        
        std::cout << "\r## Progress: " << std::setw(6) << std::setprecision(2) 
                  << progress << " % ";
    }                          
    std::cout << std::endl;
    // Output 
    std::time_t t_end = time(0);
    strftime(buf, 100, "%c", std::localtime(&t_end));
    double t_diff = std::difftime(t_end, t_start);
    log("%s: Hard work done in %.2lf s!", buf, t_diff);
    
    log("Retrieving the results ...");
    log("Creating an array to hold the mapped magnets ...");
    int *magnets = new int[phi_steps * theta_steps];
    err = clEnqueueReadBuffer(queue, magnets_cl, CL_TRUE, 0,
                              sizeof(int) * phi_steps * theta_steps, magnets,
                              0, 0, 0);
    if (err != CL_SUCCESS) {
        error(307, "Failed to retrieve magnet map: %d", err);
    }
    
    // Print the array.
    //output_magnets(magnets);
    save_image(magnets, "output.tif");
        
    // Clean up.
    log("Freeing device memory ...");
    clReleaseMemObject(coords_cl);
    clReleaseMemObject(magnets_cl);
    clReleaseMemObject(alphas_cl);
    clReleaseMemObject(rns_cl);
    delete[] coords;
    delete[] magnets;
}


int main(const int argc, const char *argv[]) {
    // Initialize OpenCL.
    cl_init();
    
    magnet_map();

    // Free used resources.
    cl_deinit();
}
