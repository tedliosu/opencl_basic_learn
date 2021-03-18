
# Instructions

1. Setup OpenCL development on your machine appropriately
2. Git clone this repository
3. Change working directory to cloned repo
4. Run "make all"
5. Run the executable that is generated by the Makefile
6. Enjoy as your GPU crunches numbers for you :) (Note - it'll take a few seconds for the data to be randomly
   generated for the calculations performed)

# Comments about code in general

- The tutorial originally had the matricies be column-major, so I rewrote it to be in row-major matricies
- As the kernel gets compiled during run-time, I left comments out of the kernel code as I do not want to break my implementation right now, but this may get changed in the future esp. if I confirm that the kernel compiler is OK with ignoring comments.


