LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib PATH=$PATH:/usr/local/cuda-10.1/bin ./cuda_memcpy 0

#sudo GIB_SRC_DIR=/home/jerrychou/libgibraltar/src  GIB_CACHE_DIR=/home/jerrychou/libgibraltar/ptx  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 PATH=$PATH:/usr/local/cuda-10.0/bin ./erasure -g 1 -w 1 -r 1 -n -d 4
#sudo GIB_SRC_DIR=/home/jerrychou/libgibraltar/src  GIB_CACHE_DIR=/home/jerrychou/libgibraltar/ptx  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 PATH=$PATH:/usr/local/cuda-10.0/bin  ./erasure -g 1 -w 1 -r 1 -n -d 1 -t $b -z 90 > output1.txt &
#sudo GIB_SRC_DIR=/home/jerrychou/libgibraltar/src  GIB_CACHE_DIR=/home/jerrychou/libgibraltar/ptx  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 PATH=$PATH:/usr/local/cuda-10.0/bin  ./erasure -g 1 -w 1 -r 1 -n -d 2 -t $b -z 90 > output2.txt &
#sudo GIB_SRC_DIR=/home/jerrychou/libgibraltar/src  GIB_CACHE_DIR=.:/home/jerrychou/gdrcopy+/ptx  LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 ./erasure -g 1 -w 1 -r 1 -n -d 2
