#include "checkpoint.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>

void save_checkpoint(const std::vector<Tensor*>& params,
    const std::string& path) {

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        printf("Checkpoint: could not open %s for writing\n", path.c_str());
        return;
    }

    for (auto param : params) {
        // Copy from GPU to CPU
        float* h_buf = new float[param->size];
        cudaMemcpy(h_buf, param->data,
            param->size * sizeof(float),
            cudaMemcpyDeviceToHost);

        // Write size then data
        fwrite(&param->size, sizeof(int), 1, f);
        fwrite(h_buf, sizeof(float), param->size, f);

        delete[] h_buf;
    }

    fclose(f);
    printf("Checkpoint saved: %s\n", path.c_str());
}

void load_checkpoint(const std::vector<Tensor*>& params,
    const std::string& path) {

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("Checkpoint: could not open %s for reading\n", path.c_str());
        return;
    }

    for (auto param : params) {
        int saved_size;
        fread(&saved_size, sizeof(int), 1, f);

        if (saved_size != param->size) {
            printf("Checkpoint: size mismatch — expected %d got %d\n",
                param->size, saved_size);
            fclose(f);
            return;
        }

        float* h_buf = new float[param->size];
        fread(h_buf, sizeof(float), param->size, f);
        cudaMemcpy(param->data, h_buf,
            param->size * sizeof(float),
            cudaMemcpyHostToDevice);
        delete[] h_buf;
    }

    fclose(f);
    printf("Checkpoint loaded: %s\n", path.c_str());
}