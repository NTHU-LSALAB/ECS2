sudo GIB_SRC_DIR=$HOMElibgibraltar/src  GIB_CACHE_DIR=$HOMElibgibraltar/ptx  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib PATH=$PATH:/usr/local/cuda-10.1/bin ./erasure -w 1 -r 1 -n -d 0 -M 4 -K 2  -m 64 -p 1