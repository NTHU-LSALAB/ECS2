sudo PATH=$PATH:/usr/local/cuda-10.0/bin  GIB_SRC_DIR=$HOME/libgibraltar/src  GIB_CACHE_DIR=$HOME/libgibraltar/ptx  LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 ./example -c $HOME/GE/ -d 0 -C ecs2file -m 4 -k 2
echo "======== File index ECS2 maintains ========="
ls $HOME/GE/
echo "===== End of File index ECS2 maintains ====="
sudo PATH=$PATH:/usr/local/cuda-10.0/bin  GIB_SRC_DIR=$HOME/libgibraltar/src  GIB_CACHE_DIR=$HOME/libgibraltar/ptx  LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 ./example -c $HOME/GE/ -d 0 -f $HOME/input -W ecs2file
echo "======== File blocks store in ECS2 ========="
ls -R /mnt
echo "===== End of File blocks store in ECS2 ====="
sudo PATH=$PATH:/usr/local/cuda-10.0/bin  GIB_SRC_DIR=$HOME/libgibraltar/src  GIB_CACHE_DIR=$HOME/libgibraltar/ptx  LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:/usr/local/lib64:/usr/lib64 ./example -c $HOME/GE/ -d 0 -f $HOME/output -R ecs2file
echo "======== Diff input output ========="
diff $HOME/input $HOME/output
echo "===== End of Diff input output ====="