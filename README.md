GPU solvers\
\
Paralution compilation:\
g++ -O3 -Wall -I../inc -c filename.cpp -o filename.o
\  
g++ -o filename filename.o -L../lib -lparalution
\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/paralution-1.1.0/build/lib/\

CuSolver compilation:\
nvcc filename.cu -o filename -lcusolver -lcusparse
