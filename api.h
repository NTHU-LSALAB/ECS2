struct GE_config {
    int dev_id;
    int buffer_size;
    CUdeviceptr d_buffer;
    uint32_t *buf_ptr;
    char* path;
};

struct GE_file {
    int m;
    int k;
    size_t size;
	size_t block_size;
    char* filename;
    char** dir_path;
    gib_context gc;
};

GE_config GE_init(int dev_id, int buffer_size, char* path);
int GE_create(GE_config config, int m, int k, char* filename, char** dir_path);
GE_file GE_open(GE_config config, char* filename);
int GE_read(GE_config config, GE_file* file, void* buffer);
int GE_pread(GE_config config, GE_file* file, void* buffer);
int GE_write(GE_config config, GE_file* file, void* buffer, size_t size);
int GE_pwrite(GE_config config, GE_file* file, void* buffer, size_t size);
int GE_close(GE_file file);