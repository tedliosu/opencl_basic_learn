
// =================================================================================================
// Project: 
// Exploring OpenCL general matrix-multiplication on an AMD GCN GPU, header file 
//
// Original file information:
// Institution.... SURFsara <www.surfsara.nl>
// Original Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
// Refactoring programmer.. Ted Li
// Changed at..... 2021-03-23
// License........ MIT license
// 
// =================================================================================================

// Range of random numbers in matrices
#define RAND_RANGE 2.0
// Smallest random number in matrices
#define SMALLEST_ENTRY -1.0
// Number of seconds to sleep for seeding random number gen function
#define SLEEP_SECS 1

// Number of OpenCL devices used
#define NUM_CL_DEVICES 1
// OpenCL Queue Properties
#define OPENCL_QUEUE_PROPERTIES 0
// Number of OpenCL Kernel programs to be loaded
#define OPENCL_KERNEL_PROGS 1
/*
 * Threadblock sizes; each size of each dimension
 * of the matricies being multiplied together MUST
 * BE divisible by the value of this macro per
 * OpenCL API specification
 */
#define TS 3
// OpenCL device name max length
#define MAX_LEN 1024
#define CL_BUFFER_OFFSET 0

// Matrix sizes
#define OPERAND_A_ROWS 3
#define OPERAND_A_COLUMNS 3
#define OPERAND_B_ROWS OPERAND_A_COLUMNS
#define OPERAND_B_COLUMNS 3
// Number of dimensions of each operand
#define OPERAND_DIMS 2

// =================================================================================================

// Set the kernel as a string
const char *kernelstring =
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    " __kernel void matrix_mulip(__global double* op_a, __global double* op_b,"
                             " __global double* op_result, int op_a_rows,"
                             " int op_a_columns, int op_b_columns)"
    "\n{"
         " const int first_dim_val = 0;"
         " const int second_dim_val = 1;"
         " const int a_row_index = get_global_id(first_dim_val);"
         " const int b_column_index = get_global_id(second_dim_val);"
         " double result_entry = 0.0;"
         " for (int a_column_index = 0; a_column_index < op_a_columns; ++a_column_index)"
         " {"
             " int b_row_index = a_column_index;"
             " result_entry += op_a[a_row_index * op_a_columns + a_column_index] * "
                               "op_b[b_row_index * op_b_columns + b_column_index];"

         " }"
         " op_result[a_row_index * op_b_columns + b_column_index] = result_entry;"
    " }";

// =================================================================================================

