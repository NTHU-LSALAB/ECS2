sudo PATH=$PATH:/usr/local/cuda-10.0/bin  GIB_SRC_DIR=/home/ron/home/jerrychou/libgibraltar/src  GIB_CACHE_DIR=/home/ron/home/jerrychou/libgibraltar/ptx  LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 ./example  -c /home/ron/home/jerrychou/GE/ -d 0 -f /home/ron/home/jerrychou/source2File -p sourceGEP -m 8 -k 4

