
// =================================================================================================
// Project: 
// Exploring OpenCL general matrix-multiplication on an AMD GCN GPU.
//
// Original file information:
// Institution.... SURFsara <www.surfsara.nl>
// Original Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Refactoring programmer.. Ted Li
// Changed at..... 2021-03-23
// License........ MIT license
// 
// =================================================================================================

// Includes and workarounds for different CL versions
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "custom_gemm_tunables.h"

// Message to user about starting calculation
#define NOTIFY_USER_CALC_START ">>> Starting calculation...\n" 

// =================================================================================================

struct Row_Maj_Matrix
{
     double* contents;
     unsigned int num_rows;
     unsigned int num_columns;
     unsigned int array_len;
};

struct Matrix_Multip_Operands
{
     struct Row_Maj_Matrix* operand_a;
     struct Row_Maj_Matrix* operand_b;
     struct Row_Maj_Matrix* operand_result;
};

struct Cl_Mem_Operands_List
{
     cl_mem* buffer_a;
     cl_mem* buffer_b;
     cl_mem* result_buffer;
};

void init_matrix(struct Row_Maj_Matrix* matrix,
                    const unsigned int num_rows,
                    const unsigned int num_columns)
{

    time_t curr_time;
    matrix->num_columns = num_columns;
    matrix->num_rows = num_rows;
    matrix->array_len = num_rows * num_columns;
    matrix->contents = (double*)malloc(num_rows*num_columns*sizeof(double*));
    time(&curr_time);
    srand(curr_time);
    sleep(SLEEP_SECS);

    for(unsigned int curr_entry = 0;
           curr_entry < matrix->array_len;
           ++curr_entry)
    {
       matrix->contents[curr_entry] = SMALLEST_ENTRY + (rand() * RAND_RANGE / RAND_MAX);
    }

}

void configure_opencl_env(cl_context *context, cl_command_queue* queue,
                                                     cl_program *program)
{
    
    cl_platform_id platform;
    cl_device_id device;
    
    clGetPlatformIDs(NUM_CL_DEVICES, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, NUM_CL_DEVICES, &device, NULL);
    *context = clCreateContext(NULL, NUM_CL_DEVICES, &device, NULL, NULL, NULL);
    *queue = clCreateCommandQueue(*context, device, OPENCL_QUEUE_PROPERTIES, NULL);
    char deviceName[MAX_LEN];
    clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_LEN, deviceName, NULL);

    // Compile the kernel
    *program = clCreateProgramWithSource(*context, OPENCL_KERNEL_PROGS, &kernelstring, NULL, NULL);
    clBuildProgram(*program, 0, NULL, "", NULL, NULL);

    // Check for compilation errors
    size_t logSize;
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    messages[logSize] = '\0';
    printf(">>> Kernel compiler result message: - %s\n", messages); 
    free(messages);

}

void prep_kernel_args(cl_context *context, cl_command_queue* queue,
                        cl_program *program, cl_kernel* kernel,
                        struct Matrix_Multip_Operands operands,
                        struct Cl_Mem_Operands_List cl_operands)
{

    // Prepare OpenCL memory objects
    *(cl_operands.buffer_a) = clCreateBuffer(*context, CL_MEM_READ_ONLY, 
                                                operands.operand_a->array_len*sizeof(double), NULL, NULL);
    *(cl_operands.buffer_b) = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                                operands.operand_b->array_len*sizeof(double), NULL, NULL);
    *(cl_operands.result_buffer) = clCreateBuffer(*context, CL_MEM_READ_WRITE,
                                                     operands.operand_result->array_len*sizeof(double), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(*queue, *(cl_operands.buffer_a), CL_TRUE, CL_BUFFER_OFFSET,
                                         operands.operand_a->array_len*sizeof(double),
                                            operands.operand_a->contents, 0, NULL, NULL);
    clEnqueueWriteBuffer(*queue, *(cl_operands.buffer_b), CL_TRUE, CL_BUFFER_OFFSET,
                                          operands.operand_b->array_len*sizeof(double),
                                            operands.operand_b->contents, 0, NULL, NULL);
    clEnqueueWriteBuffer(*queue, *(cl_operands.result_buffer), CL_TRUE, CL_BUFFER_OFFSET,
                                         operands.operand_result->array_len*sizeof(double),
                                          operands.operand_result->contents, 0, NULL, NULL);

    // Configure the kernel and set its arguments
    *kernel = clCreateKernel(*program, "matrix_mulip", NULL);
    clSetKernelArg(*kernel, 0, sizeof(cl_mem), (void*)cl_operands.buffer_a);
    clSetKernelArg(*kernel, 1, sizeof(cl_mem), (void*)cl_operands.buffer_b);
    clSetKernelArg(*kernel, 2, sizeof(cl_mem), (void*)cl_operands.result_buffer);
    clSetKernelArg(*kernel, 3, sizeof(unsigned int), (void*)&(operands.operand_a->num_rows));
    clSetKernelArg(*kernel, 4, sizeof(unsigned int), (void*)&(operands.operand_a->num_columns));
    clSetKernelArg(*kernel, 5, sizeof(unsigned int), (void*)&(operands.operand_b->num_columns));

}

void print_matrix(struct Row_Maj_Matrix* matrix) {

    for (unsigned int curr_row = 0; curr_row < matrix->num_rows; ++curr_row)
    {
        for (unsigned int curr_column = 0; curr_column < matrix->num_columns; ++curr_column)
        { 
            printf("%f ", matrix->contents[curr_row * matrix->num_columns + curr_column]);
       
        }
        printf("\n");
    }
    
    printf("\n");

}

// Matrix-multiplication using a custom OpenCL SGEMM kernel.
int main(int argc, char* argv[]) {
    
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer_a;
    cl_mem buffer_b;
    cl_mem result_buffer;
    cl_event event;
    struct Row_Maj_Matrix operand_a;
    struct Row_Maj_Matrix operand_b;
    struct Row_Maj_Matrix operand_result;
    
    struct Cl_Mem_Operands_List cl_mem_ops =
               { &buffer_a, &buffer_b, &result_buffer };
    struct Matrix_Multip_Operands matrix_ops =
               { &operand_a, &operand_b, &operand_result };
    // Set Matrix Sizes
    init_matrix(&operand_a, OPERAND_A_ROWS, OPERAND_A_COLUMNS);
    init_matrix(&operand_b, OPERAND_B_ROWS, OPERAND_B_COLUMNS);
    init_matrix(&operand_result, OPERAND_A_ROWS, OPERAND_B_COLUMNS);
    
    configure_opencl_env(&context, &queue, &program);

    prep_kernel_args(&context, &queue, &program, &kernel, matrix_ops, cl_mem_ops);

    // Notify user
    printf(NOTIFY_USER_CALC_START);

    // Run the kernel
    const size_t local[OPERAND_DIMS] = { TS, TS };
    const size_t global[OPERAND_DIMS] = { operand_a.num_rows, operand_b.num_columns };
    clEnqueueNDRangeKernel(queue, kernel, OPERAND_DIMS, NULL, global, local, 0, NULL, &event);

    // Wait for calculations to be finished
    clWaitForEvents(1, &event);

    // Copy the output matrix C back to the CPU memory
    clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, CL_BUFFER_OFFSET, operand_result.array_len*sizeof(double),
                                                                           operand_result.contents, 0, NULL, NULL);

    print_matrix(&operand_a);
    print_matrix(&operand_b);
    print_matrix(&operand_result);

    // Free the OpenCL memory objects
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(result_buffer);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Free the host memory objects
    free(operand_a.contents);
    free(operand_b.contents);
    free(operand_result.contents);

    // Exit
    return EXIT_SUCCESS;
}

// =================================================================================================
