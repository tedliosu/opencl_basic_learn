
# Instructions

1. Setup OpenCL development on your machine appropriately
   1. Refer to [these set of instructions](https://github.com/tedliosu/opencl_install_instructions)
      if **running a Linux distro** on an AMD GPU and/or any kind of CPU.
   2. Next (if on a Linux distro), make sure your "CPPFLAGS" and "LDFLAGS" environment variables are set appropriately. (e.g. CPPFLAGS="-I/opt/rocm/opencl/include" LDFLAGS="-L/opt/rocm/opencl/lib").
2. Git clone this repository
3. Change working directory to cloned repo
4. Run "make all"
5. Run the executable that is generated by the Makefile
6. Enjoy as your OpenCL device crunches numbers for you :) (Note - it'll take a few seconds for the data to be randomly
   generated for the calculations performed)

# Comments about code in general

- The tutorial originally had the matricies be column-major, so I rewrote it to be in row-major matricies
- ~~As the kernel gets compiled during run-time, I left comments out of the kernel code as I do not want to break my implementation right now, but this may get changed in the future esp. if I confirm that the kernel compiler is OK with ignoring comments.~~  Comments added on 6/24.


