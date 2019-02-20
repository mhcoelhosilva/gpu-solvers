#include<stdio.h>
#include<time.h>
#include<cuda.h>
#include<cusolverSp.h>
#include<cusparse.h>
#include<math.h>

void printRunTime(const char *string, struct timespec *ti1, struct timespec *ti2)
{
   double runtime;
   
   runtime = (ti2->tv_sec - ti1->tv_sec ) + 1e-9*(ti2->tv_nsec - ti1->tv_nsec);
   fprintf(stderr,"Run time %s : %lf secs.\n", string, runtime);
   
}


int main(int argc, char * argv[]) 
{
    
    int m, sizeA;
    
    m = 301; /* preferable take odd m */
    double h = 1.0/(m+1.0);
    sizeA = m*m;
    
    
    
    int nnz = 5*m*m-2*(1+m);
    /* store in CSR format (not COO) */
    int* I = (int *)malloc((sizeA+1)*sizeof(int)); 
    int* J = (int *)malloc(nnz*sizeof(int));
    double* V = (double *)malloc(nnz*sizeof(double));
    
    int nzCounter = 0;
    
    for (int i = 0; i < sizeA; ++i) {
        I[i] = nzCounter;
        for (int j = 0; j < sizeA; ++j) {
            if (i == j) {
                V[nzCounter] = 4*(1.0/(h*h));
                J[nzCounter] = j;
                nzCounter++;
            } else if (j == i + 1) {
                V[nzCounter] = -1*(1.0/(h*h));
                J[nzCounter] = j;
                nzCounter++;
            } else if (j == i - 1) {
                V[nzCounter] = -1*(1.0/(h*h));
                J[nzCounter] = j;
                nzCounter++;
            } else if (j == i + m) {
                V[nzCounter] = -1*(1.0/(h*h));
                J[nzCounter] = j;
                nzCounter++;
            } else if (j == i - m) {
                V[nzCounter] = -1*(1.0/(h*h));
                J[nzCounter] = j;
                nzCounter++;
            }
            
        } 
    }
    
    I[sizeA] = nzCounter;
    
    

    struct timespec ti1,ti2;   
    
    
    double* b = (double *)malloc(sizeA*sizeof(double));
    double* x = (double *)malloc(sizeA*sizeof(double));
    double* r = (double *)malloc(sizeA*sizeof(double));
    for (int i=0; i < sizeA; i++)
    {
        b[i] = 0;
        x[i] = 0;
    }
    
    int middle = (m*m-1)/2;;
    b[middle] = 1.0/(h*h);
    
    
    clock_gettime(CLOCK_REALTIME,&ti1);  
    
    double* d_V = NULL;
    int* d_I = NULL;
    int* d_J = NULL;
    double* d_b = NULL;
    double* d_x = NULL;
    
    
    cudaMalloc(&d_V, nnz*sizeof(double));
    cudaMalloc(&d_I, (sizeA+1)*sizeof(int));
    cudaMalloc(&d_J, nnz*sizeof(int));
    cudaMalloc(&d_b, sizeA*sizeof(double));
    cudaMalloc(&d_x, sizeA*sizeof(double));
    
    
    cudaMemcpy(d_V, V, nnz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, I, (sizeA+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_J, J, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeA*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeA*sizeof(double), cudaMemcpyHostToDevice);
    
    
    int singularity = 0;
    int reorder = 0;
    double tol = 1.0e-11;
    cusolverSpHandle_t handle;
    cusolverStatus_t status;
    status = cusolverSpCreate(&handle);
    cusparseMatDescr_t descrA;
    cusparseStatus_t sparse_status;
    sparse_status = cusparseCreateMatDescr(&descrA);
    
    
    cudaDeviceSynchronize();
   
      
    
    
    status = cusolverSpDcsrlsvqr( 
                  handle,
                  sizeA,
                  nnz,
                  descrA,
                  d_V,
                  d_I,
                  d_J,
                  d_b,
                  tol,
                  reorder,
                  d_x,
                  &singularity);
    
    cudaDeviceSynchronize();
     
    
    clock_gettime(CLOCK_REALTIME,&ti2); 
    
    cudaMemcpy(x, d_x, sizeA*sizeof(double),cudaMemcpyDeviceToHost);
    
    cudaFree(d_V);
    cudaFree(d_I);
    cudaFree(d_J);
    cudaFree(d_b);
    cudaFree(d_x);
    
    
    printRunTime("runtime cuSolver",&ti1,&ti2);
    
    /*
    int xx,yy;
    
    
    for(int i = 0; i< sizeA; ++i)
    {
        xx = i%m;
        yy = (i-xx)/m;
        printf("%d %d %lf \n", xx, yy, x[i]); 
    } */
    
    
    for(int i = 0; i< sizeA; i++)
        r[i] = 0;
    
    for(int i = 0; i < sizeA; i++)
        for( int k = I[i]; k < I[i+1]; k++)
        {
            r[i] = r[i] + V[k]*x[J[k]];  
        }
    r[middle] = r[middle] - b[middle];
    
    
    
    
    
    
    double L2r = 0.0;
    double maxres = 0.0;
    for (int i=0;i<sizeA;i++)
    {
        L2r += r[i]*r[i];
        if (abs(r[i])>maxres)
        {maxres = abs(r[i]);}
    }
    
    L2r = (1.0/(double)sizeA)*sqrt(L2r);
    

    printf("our L2 residual = %.16lf \n", L2r);
    printf("our max residual = %.16lf \n", maxres);
    
    
    free(V);
    free(I);
    free(J);
    free(b);
    free(x);
    free(r);
    
    
    return 0;
}
/*nvcc poisson_cuSolver_2d.cu -o poisson_cuSolver_2d -lcusolver -lcusparse */
