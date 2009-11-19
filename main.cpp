/*
 *
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
#include <OpenCL/opencl.h>
#include <IL/il.h>

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const char *kernel_filename = "magnetic_pendulum.cl";
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
const int phi_steps = 32, theta_steps = 32;
ILubyte colors[3*3] = {255, 0, 0,
					   0, 255, 0,
					   0, 0, 255 };

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
    kernel = clCreateKernel(program, "map_magnets", &err);
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
	// Create image.
	ILubyte *pixels = new ILubyte[phi_steps * theta_steps * 3];
	for (int y = 0; y < theta_steps; ++y) {
        for (int x = 0; x < phi_steps; ++x) {
			int magnet = magnets[y * phi_steps + x];
			int index = 3 * (phi_steps * (theta_steps - y - 1) + x);
			pixels[index + 0] = colors[3 * magnet + 0];
			pixels[index + 1] = colors[3 * magnet + 1];  
			pixels[index + 2] = colors[3 * magnet + 2];
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
	
	// Create an array holding the mapped magnets.
	size_t magnets_len = phi_steps * theta_steps;
	int *magnets = new int[magnets_len];
	
	// Get device memory.
	cl_mem coords_cl;
	cl_mem magnets_cl;
	coords_cl = clCreateBuffer(context, CL_MEM_READ_ONLY,
							   sizeof(float) * coords_len, 0, 0);
	magnets_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
								sizeof(int) * magnets_len, 0, 0);
	if (!magnets_cl || !coords_cl) {
		error(301, "Could not allocate device memory!");
	}
	
	// Write coordinates.
	err = clEnqueueWriteBuffer(queue, coords_cl, CL_TRUE, 0,
							   sizeof(float) * coords_len, coords,
							   0, 0, 0);
	if (err != CL_SUCCESS) {
		error(302, "Could not write to device memory.");
	}
	
	// Set compute kernel arguments.
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &coords_cl);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &magnets_cl);
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &phi_steps);
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &theta_steps);
	if (err != CL_SUCCESS) {
		error(303, "Could no set kernel parameters: %d", err);
	}
	
	// Retrieve work group sizes.
	size_t global, local;
	err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
								   sizeof(size_t), &local, 0);
	if (err != CL_SUCCESS) {
		error(304, "Failed to retrieve kernel work group info: %d", err);
	}
	
	// Execute kernel over the entire range and using the maximum work group
	// items.
	global = magnets_len;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, 0, &global, &local,
								 0, 0, 0);
	if (err) {
		error(305, "Failed to execute kernel: %d", err);
	}
	
	// Wait for the kernel to finish.
	clFinish(queue);
	
	// Retrieve the magnet map.
	err = clEnqueueReadBuffer(queue, magnets_cl, CL_TRUE, 0,
							  sizeof(int) * magnets_len, magnets,
							  0, 0, 0);
	if (err != CL_SUCCESS) {
		error(306, "Failed to retrieve magnet map.");
	}
	
	// Print the array.
	//output_magnets(magnets);
	save_image(magnets, "output.tif");
		
	// Clean up.
	clReleaseMemObject(coords_cl);
	clReleaseMemObject(magnets_cl);
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
