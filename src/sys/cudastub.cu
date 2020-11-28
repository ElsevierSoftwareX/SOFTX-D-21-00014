/****************************************************************************
** 
**  Copyright (C) 2019-2020 Boris Krasnopolsky, Alexey Medvedev
**  Contact: xamg-test@imec.msu.ru
** 
**  This file is part of the XAMG library.
** 
**  Commercial License Usage
**  Licensees holding valid commercial XAMG licenses may use this file in
**  accordance with the terms of commercial license agreement.
**  The license terms and conditions are subject to mutual agreement
**  between Licensee and XAMG library authors signed by both parties
**  in a written form.
** 
**  GNU General Public License Usage
**  Alternatively, this file may be used under the terms of the GNU
**  General Public License, either version 3 of the License, or (at your
**  option) any later version. The license is as published by the Free 
**  Software Foundation and appearing in the file LICENSE.GPL3 included in
**  the packaging of this file. Please review the following information to
**  ensure the GNU General Public License requirements will be met:
**  https://www.gnu.org/licenses/gpl-3.0.html.
** 
****************************************************************************/

#include <stdio.h>
#include <dlfcn.h>


extern "C" { 

void cuda_stub_setup() {
}
cudaError_t CUDARTAPI cudaConfigureCall(dim3 g, dim3 b, size_t sharedMem, cudaStream_t stream)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaDeviceSynchronize(void)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);    
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *p, const void *c)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return NULL;

}
cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaHostUnregister(void *ptr)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaLaunch(const void *func)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaSetDevice(int device)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);   
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}
cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;

}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t Stream)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}

void** CUDARTAPI  __cudaRegisterFatBinary(void *fatCubin)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return 0;

}
void CUDARTAPI __cudaRegisterFatBinaryEnd(void **)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return;
}
void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return;

}
void CUDARTAPI __cudaRegisterFunction(void   **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return;

}

char CUDARTAPI __cudaInitModule(void **fatCubinHandle) 
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return 0;
}

cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) 
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) 
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
{
#ifdef WITH_GPU_VERBOSE    
    printf(">> STUB: %s\n", __FUNCTION__);
#endif
    return cudaSuccess;
}

}
