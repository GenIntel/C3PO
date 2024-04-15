```

wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
chmod +x cuda_11.7.0_515.43.04_linux.run
./cuda_11.7.0_515.43.04_linux.run

mv /home/sommerl/Downloads/cudnn-linux-x86_64-8.9.5.29_cuda11-archive.tar.xz .
tar -xf cudnn-linux-x86_64-8.9.5.29_cuda11-archive.tar.xz
CUDNN=cudnn-linux-x86_64-8.9.5.29_cuda11-archive
CUDA_HOME=/scratch/sommerl/cudas/cuda-11.7
cp ${CUDNN}/include/cudnn*.h ${CUDA_HOME}/include
cp ${CUDNN}/lib/libcudnn* ${CUDA_HOME}/lib64
chmod a+r ${CUDA_HOME}/include/cudnn*.h ${CUDA_HOME}/lib64/libcudnn*


/work/dlclarge1/sommerl-od3d/cudas/cuda-11.7

CUDA_HOME=/scratch/sommerl/cudas/cuda-11.7
PATH=${PATH}:${CUDA_HOME}/bin
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
export LD_LIBRARY_PATH
export PATH

export CUDA_HOME

```
