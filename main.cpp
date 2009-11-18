#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cstdarg>
#include <cstdlib>
#include <cmath>
#include <OpenCL/opencl.h>

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------

const char *kernel_filename = "magnetic_pendulum.cl";
const bool use_gpu = true;

// --------------------------------------------------------------------------
// Globals
// --------------------------------------------------------------------------

int err;                // Error code
cl_device_id device_id; // Compute device ID
cl_context context;     // Compute context
cl_command_queue queue; // Command queue
cl_program program;     // Compute program
cl_kernel kernel;       // Compute kernel


// --------------------------------------------------------------------------
// Functions
// --------------------------------------------------------------------------

/*
 * Output an error message and quit.
 *
 */
void error(int code, const char *format, ...) {
    char buf[256];
    va_list args;

    va_start(args, format);
    std::vsprintf(buf, format, args);
    std::cerr << "Error " << code << ": " << buf << std::endl;
    va_end(args);

    std::exit(code);
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
    // Connect to a compute device.
    err = clGetDeviceIDs(0, use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU,
                         1, &device_id, 0);
    if (err != CL_SUCCESS)
        error(200, "Failed to create a device group!");

    // Create a compute context.
    context = clCreateContext(0, 1, &device_id, 0, 0, &err);
    if (!context)
        error(201, "Failed to create a compute context!");

    // Create a command queue.
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue)
        error(202, "Failed to create a command queue!");

    // Create a compute program from file.
    std::string source = file_content(kernel_filename);
    const char *str = source.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) &str,
                                        0, &err);
    if (!program)
        error(203, "Failed to create compute program!");

    // Build the program executable.
    err = clBuildProgram(program, 0, 0, "-Werror", 0, 0);
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
        std::cout << "Program size: " << program_size << " bytes" << std::endl;
    }

    // Create the compute kernel.
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
        error(205, "Failed to create compute kernel");
}

/*
 * Free OpenCL resources that were previously allocated. 
 *
 */
void cl_deinit() {
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}



int main(const int argc, const char *argv[]) {
    cl_init();


    cl_deinit();
}
