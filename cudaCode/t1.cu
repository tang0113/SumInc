#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "iostream"

__global__ void sayHelloWorld();

int main(){

    int dev = 0;
    cudaDeviceProp devProp;
    (cudaGetDeviceProperties(&devProp, dev));
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    // sayHelloWorld<<<1, 10>>>();   //调用GPU上执行的函数，调用10个GPU线程
    // printf("HelloWorld! CPU \n");
    // cudaError_t cudaStatus = cudaGetLastError();
	// if (cudaStatus != cudaSuccess)
	// {
	// 	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	// }
    // cudaDeviceReset();    //显式地释放和清空当前进程中与当前设备有关的所有资源，不加这句不会打印GPU中的输出语句"HelloWorld! GPU"

    // system("pause");
    return 0;
}

__global__ void sayHelloWorld(){
    printf("HelloWorld! GPU \n");
    //cout << "HelloWorld! GPU" << endl;     //不能使用cout, std命名不能使用到GPU上
}