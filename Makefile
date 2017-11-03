CC=nvcc
CCFLAGS=--compiler-options=-Wall -g
LDFLAGS=-lm
SRC=main.cu

fft: $(SRC) argparse.o
	$(CC) $(CCFLAGS) argparse.o $< -o $@ $(LDFLAGS)

argparse.o: argparse.c
	$(CC) $(CCFLAGS) -c $< 

clean:
	rm -f fft argparse.o
