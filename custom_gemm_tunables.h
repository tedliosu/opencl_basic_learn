
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
         "\n/* "
         "   * Values representing first and second work-item dimensions in OpenCL;"
         "   * two dimensions are needed as the data structures we're working with"
         "   * are matrices which are two dimensional data structures."
         "   */\n"
         " const int first_dim_val = 0, second_dim_val = 1;"
         "\n/* "
         "   * Map each row of the first matrix (after pairing with each column"
         "   * of the second matrix) to an instance of this kernel function"
         "   * for evenly dividing up the work of multiplying two matricies"
         "   * together; each instance of this kernel function is responsible"
         "   * for generating one element in the resulting matrix."
         "   */\n"
         " const int a_row_index = get_global_id(first_dim_val), "
                   " b_column_index = get_global_id(second_dim_val);"
         " double result_entry = 0.0;"
         "\n/* "
         "   * Loop through each column of the first matrix and each row of the"
         "   * second matrix at the same time according to the definition"
         "   * of matrix multiplication and multiply each pair of numbers"
         "   * encountered together."
         "   */\n"
         " for (int a_column_index = 0; a_column_index < op_a_columns; ++a_column_index)"
         " {"
             " int b_row_index = a_column_index;"
             " result_entry += op_a[a_row_index * op_a_columns + a_column_index] * "
                               "op_b[b_row_index * op_b_columns + b_column_index];"

         " }"
         " // Store each result element in the output matrix. \n"
         " op_result[a_row_index * op_b_columns + b_column_index] = result_entry;"
    " }";

// =================================================================================================

