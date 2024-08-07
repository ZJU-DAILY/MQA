//
// Created by MaouSanta on 2024/2/17.
//

#ifndef INDEXING_AND_SEARCH_META_VECTOR_H
#define INDEXING_AND_SEARCH_META_VECTOR_H

#include <fstream>
#include "../data_objects_define.h"

template<typename T = float>
struct MetaVector {
    std::vector<std::vector<T>> vec{};
    size_t num = 0;
    std::string file_path;

    CStatus norm(std::vector<std::vector<T>> v, const size_t n) {
        for (size_t i = 0; i < n; i++) {
            float vector_norm = 0;
            unsigned d = v[i].size();
            for (size_t j = 0; j < d; j++) {
                vector_norm += v[i][j] * v[i][j];
            }
            vector_norm = std::sqrt(vector_norm);
            for (size_t j = 0; j < d; j++) {
                v[i][j] /= vector_norm;
            }
        }
        return CStatus();
    }

    CStatus load(const std::string& path, const unsigned is_norm) {
        std::ifstream in(path.data(), std::ios::binary);
        if (!in.is_open()) {
            return CStatus(path + " open file error!");
        }
        unsigned dim_val_size = sizeof(unsigned);
        unsigned dim;
        while (!in.eof()) {
            in.read((char *) &dim, dim_val_size);
            if (in.eof()) break;
            std::vector<IDType> tmp(dim);
            in.read((char *) tmp.data(), dim * sizeof(IDType));
            vec.emplace_back(tmp);
        }
        in.close();
        num = vec.size();
        file_path = path;
        if (is_norm) {
            norm(vec, num);
            printf("[EXEC] normalize vector complete!\n");
        }
        return CStatus();
    }

    virtual ~MetaVector() {
        if (!vec.empty()) {
            for (IDType i = 0; i < vec.size(); i++) {
                std::vector<IDType>().swap(vec[i]);
            }
            std::vector<std::vector<T>>().swap(vec);
        }
    }
};

#endif //INDEXING_AND_SEARCH_META_VECTOR_H
