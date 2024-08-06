//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_META_DATA_H
#define INDEXING_AND_SEARCH_META_DATA_H

#include "../data_objects_define.h"

template<typename T = float>
struct MetaData {
    T *data = nullptr;
    IDType num = 0;
    unsigned dim = 0;
    std::string file_path;

    CStatus norm(T *vec, const unsigned n, const unsigned d) {
        for (size_t i = 0; i < n; i++) {
            float vector_norm = 0;
            for (size_t j = 0; j < d; j++) {
                vector_norm += vec[i * d + j] * vec[i * d + j];
            }
            vector_norm = std::sqrt(vector_norm);
            for (size_t j = 0; j < d; j++) {
                vec[i * d + j] /= vector_norm;
            }
        }
        return CStatus();
    }

    CStatus load(const std::string& path, const unsigned is_norm = 1) {
        std::ifstream in(path.data(), std::ios::binary);
        if (!in.is_open()) {
            return CStatus(path + " open file error!");
        }
        unsigned dim_val_size = sizeof(unsigned);
        in.read((char *) &dim, dim_val_size);
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        auto f_size = (size_t) ss;
        num = (IDType) (f_size / (dim * sizeof(T) + dim_val_size));
        data = new T[(size_t)num * (size_t)dim];

        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in.seekg(dim_val_size, std::ios::cur);
            in.read((char *) (data + i * dim), dim * sizeof(T));
        }
        in.close();
        file_path = path;
        if (is_norm) {
            norm(data, num, dim);
            printf("[EXEC] normalize vector complete!\n");
        }
        return CStatus();
    }

    virtual ~MetaData() {
        if (data) {
            std::cout << "[DEBUG] modal data release: " << data << std::endl;
            delete[] data;
            data = nullptr;
        }
    }
};

#endif //INDEXING_AND_SEARCH_META_DATA_H
