#include<iostream>
#include<paralution.hpp>
#include<cmath>
#includeZ<iomanip>

using namespace paralution;



int main(int argc, char * argv[]) 
{
    init_paralution();
    
    int m, sizeA, xxx, yyy;
    
    int k = 0;
    
    double xx,yy;
    
    m = 801; /*m = (N-1), i.e. without including boundary points. Dirichlet b.c. for now, u = 0 at boundary.*/
    int nnz = 5*m*m-2*(1+m)-(m-1)*2 +4*m + 4*(m+1); 
    m = m + 2;
    double h = 1.0/(m-1);
    sizeA = m*m;
    LocalMatrix<double> A;
    LocalVector<double> b,x,r;
    
    int* I = new int[nnz];
    int* J = new int[nnz];
    double* V = new double[nnz];
    
    int nzCounter = 0;
    
    b.Allocate("b", sizeA);
    b.Zeros();
    
    
    for (int i = 0; i < sizeA; ++i) {
        for (int j = 0; j < sizeA; ++j) {
           xxx = i%m;
           xx = (i%m)*h;
           yy = ((i-xxx)/m)*h;
            if (i == j) {
                if (j < m ) {
                    b[i] = sin(xx);
                    V[nzCounter] = 1;
                    I[nzCounter] = i;
                    J[nzCounter] = j;
                    nzCounter++;
                } else if (j % m == 0 ) {
                    b[i] = 0;
                    V[nzCounter] = 1;
                    I[nzCounter] = i;
                    J[nzCounter] = j;
                    nzCounter++;
                } else if (j > m*(m-1)) {
                    b[i] = sin(xx)*cos(1.0);
                    V[nzCounter] = 1;
                    I[nzCounter] = i;
                    J[nzCounter] = j;
                    nzCounter++;
                } else if ((j+1) % m == 0) {
                    b[i] = sin(1.0)*cos(yy);
                    V[nzCounter] = 1;
                    I[nzCounter] = i;
                    J[nzCounter] = j;
                    nzCounter++;
                } else {
                    V[nzCounter] = 4*(m-1)*(m-1) - k*k;
                    I[nzCounter] = i;
                    J[nzCounter] = j;
                    b[i] = (2-k*k)*sin(xx)*cos(yy);
                    nzCounter++;
                }
            } else if (!((i < m ) || (i % m == 0 ) || (i > m*(m-1)) || ((i+1) % m == 0))) {    
              if ((j == i + 1) /*&& (j % m != 0)) */ ) {
                V[nzCounter] = -1*(m-1)*(m-1);
                I[nzCounter] = i;
                J[nzCounter] = j;
                
                nzCounter++;
            } else if ((j == i - 1) /*&& (j == 0 || i % m != 0))*/ ){
                V[nzCounter] = -1*(m-1)*(m-1);
                I[nzCounter] = i;
                J[nzCounter] = j;
                
                nzCounter++;
            } else if (j == i + m) {
                V[nzCounter] = -1*(m-1)*(m-1);
                I[nzCounter] = i;
                J[nzCounter] = j;
                
                nzCounter++;
            } else if (j == i - m) {
                V[nzCounter] = -1*(m-1)*(m-1);
                I[nzCounter] = i;
                J[nzCounter] = j;
                
                nzCounter++;
            } 
        }
    }
    }

   
    double tick, tack;
    tick = paralution_time();
   
    A.Assemble(I,J,V,nnz,"A", sizeA, sizeA);
   
   
    x.Allocate("x", sizeA);
    x.Zeros();
    
    r.Allocate("r", sizeA);
    
    A.MoveToAccelerator();
    b.MoveToAccelerator();
    x.MoveToAccelerator();
    r.MoveToAccelerator();
    
    GMRES<LocalMatrix<double>,LocalVector<double>,double>it;
    AMG<LocalMatrix<double>,LocalVector<double>,double>p;
  
    
    it.RecordResidualHistory();
    
    it.InitMaxIter(1000000000);
    
    it.SetPreconditioner(p);
    it.SetOperator(A);
    it.Build();
    
    
    
    it.Solve(b,&x);
    tack = paralution_time();
    
    A.Apply(x,&r);
    r.AddScale(b,-1);
    double residual, maxres;
    r.Amax(maxres);
    residual = r.Norm();
    
    r.MoveToHost();
    x.MoveToHost();
    
    residual = (1.0/m)*residual;
    std::cout << "L2 residual =" << residual << " max residual " << maxres << std::endl; 
    std::cout << "solve time =" << (tack-tick)/1000000 << std::endl;
    
    
    double* Uexact = new double[sizeA];
    double* error = new double[sizeA];
    
    double errorL2 = 0.0;
    double errormax = 0.0;

    
    for(int i = 0; i< sizeA; ++i)
    {
       
        xxx = i%m;
        xx = (i%m)*h;
        yy = ((i-xxx)/m)*h;
        Uexact[i] = sin(xx)*cos(yy); 
        error[i] = fabs(x[i] - Uexact[i]);
        errorL2 += (error[i])*(error[i]);
        if (error[i]>errormax){errormax=error[i];}
        
        
    }
    
    
    std::cout << std::fixed;
    std::cout << std::setprecision(16);
    
    errorL2 = (1.0/(double)sizeA)*sqrt(errorL2);
    
   
    std::cout << "max error =" << errormax << " L2 error =" << errorL2 << std::endl;
    
    /*
    for(int i = 0; i< sizeA; ++i)
    {
        xxx = i%m;
        yyy = (i-xxx)/m;
        std::cout << xxx <<" " << yyy << " " << x[i] <<  std::endl; 
    } */
    
    b.Clear();
    A.Clear();
    x.Clear();
    
    stop_paralution();
    
    return 0;
}

/*
k=10, m=201 : 3700945
k=10, m=301 : 3700946
k=10, m=501 : 3700947
*/
