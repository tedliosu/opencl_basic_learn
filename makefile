
main_c_file := $(wildcard *.c) 
main_prog_file = $(basename $(main_c_file)) 

all: $(main_c_file)
	gcc -g -o $(main_prog_file) $? $(CPPFLAGS) -lOpenCL $(LDFLAGS)

clean:
	rm -f $(main_prog_file)

