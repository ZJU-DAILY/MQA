# MQA

MQA is an interactive Multi-modal Query Answering system, powered by [MUST](https://github.com/ZJU-DAILY/MUST) and latest LLMs. It comprises five core components: Data Preprocessing, Vector Representation, Index Construction, Query Execution, and Answer Generation, all orchestrated by a dedicated coordinator to ensure smooth data flow from input to answer generation.

## Requirements

We make use of the [Anaconda](https://www.anaconda.com/) package manager in order to avoid dependency/reproducibility problems. 

1. Clone the repository
   ```
   git clone https://github.com/ZJU-DAILY/MQA
   ```

2. Install Python dependencies using conda.

   ```
   conda create --name mqa --file environment.yml
   conda activate mqa
   ```

   Or install them manually.
   ```
   conda create -n mqa -y python=3.11.5
   conda activate mqa
   conda install flask=2.2.5
   conda install -y -c pytorch pytorch=2.1.2 torchvision=0.16.2
   pip install openai==1.14.0
   pip install openai-clip
   ```

3. Compile C++ code for indexing and searching.

   ```
   cd ./indexing_and_search
   git clone https://github.com/ChunelFeng/CGraph.git
   cmake build
   make
   ```

4. install npm and nodejs>=18.0.

5. To make use of OpenAI's LLMs, please set up your [API key](https://platform.openai.com/docs/quickstart?context=python) first.

## Quick start

1. Launch the Flask server as the backend.

   ```
   python app.py
   ```

2. Launch the frontend in another terminal instance.
   ```
   cd ./frontend
   npm install
   npm run dev
   ```

## Usage

### Data Preparation

To properly work with the MIT-States dataset, the following structure is required:

```
MQA_base_path
├─dataset
│  ├─base
│  ├─meta
│  ├─MitStates
│  │  └─images
│  │      ├─adj aluminum
│  │      ├─adj animal
│  │      ├─adj apple
│  │      ├─...
```

### Pre-processed data

We provide the pre-processed data via [Google Drive](https://drive.google.com/drive/folders/1tFt04JjjYdScCpCKYrRsxO6gAbbY8t9s?usp=sharing) in case you don't have enough GPU resources or simply want to save time. Download them and move them to `/dataset/base` and `/dataset/meta` as shown in the directory above. These data will also be created during the use of MQA.

## Citation

```BibTeX
@manual{MQA,
  author    = {Mengzhao Wang and Haotian Wu and Xiangyu Ke and Yunjun Gao and Xiaoliang Xu and Lu Chen},
  title     = {An Interactive Multi-modal Query Answering System with Retrieval-Augmented Large Language Models},
  url       = {https://github.com/ZJU-DAILY/MQA},
  year      = {2024}
}

@inproceedings{MUST_ICDE24,
title={{MUST}: An Effective and Scalable Framework for Multimodal Search of Target Modality},
author={Mengzhao Wang and Xiangyu Ke and Xiaoliang Xu and Lu Chen and Yunjun Gao and Pinpin Huang and Runkai Zhu},
booktitle={IEEE International Conference on Data Engineering (ICDE)},
year={2024}
}

@manual{MVG_VLDB2024,
  author    = {Mengzhao Wang and Xiangyu Ke and Lu Chen and Yunjun Gao},
  title     = {MVG Index: Empowering Multi-Vector Similarity Search in High-Dimensional Spaces},
  url       = {https://github.com/ZJU-DAILY/MVG/blob/main/MVG_technical_report.pdf},
  year      = {2024}
}
```

## Acknowledgement

+ [CGraph](https://github.com/ChunelFeng/CGraph): A cross-platform Directed Acyclic Graph framework based on pure C++ without any 3rd-party dependencies.
