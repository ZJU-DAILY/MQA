# MQA

MQA is a interactive Multi-modal Query Answering system, powered by [MUST](https://github.com/ZJU-DAILY/MUST) and [CGraph](https://github.com/ChunelFeng/CGraph) and latest LLMs. Comprising five primary components - Data Preprocessing, Vector Representation, Index Construction, Query Execution, Answer Generation - MQA possesses a modular configuration.

## Requirements

We make the use of the [**Anaconda**](https://www.anaconda.com/) package manager in order to avoid dependency/reproducibility problems. 

1. Clone the repository
   ```
   git clone https://github.com/ZJU-DAILY/MQA
   ```

2. Install Python dependencies with conda.

   ```
   conda create --name mqa --file environment.yml
   conda activate mqa
   ```

   Or install them manually
   ```
   conda create -n mqa -y python=3.11.5
   conda activate mqa
   conda install flask=2.2.5
   conda install -y -c pytorch pytorch=2.1.2 torchvision=0.16.2
   pip install openai==1.14.0
   pip install openai-clip
   ```

3. Compile C++ code for indexing and searching, and move the executable file to 

   ```
   cd ./indexing_and_search
   git clone https://github.com/ChunelFeng/CGraph.git
   cmake build
   make
   ```

4. install npm and nodejs>=18.0

5. To make use of OpenAI's LLMs, please set up your [API key](https://platform.openai.com/docs/quickstart?context=python) first

## Quick start

1. Launch Flask server as backend

   ```
   python app.py
   ```

2. Launch frontend in another terminal instance
   ```
   cd ./frontend
   npm install
   npm run dev
   ```

## Usage

### Data Preparation

To properly work with the codebase MIT-States datasets should have the following structure:

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

We provide the pre-processed data via [Google Drive](https://drive.google.com/drive/folders/1tFt04JjjYdScCpCKYrRsxO6gAbbY8t9s?usp=sharing) in case you don't have enough GPU resources or just want to save some time, download them and remove to `/dataset/base` and `/dataset/meta` just as the showed directory above. These data will also be created during the use of MQA.

## Citation

```BibTeX
@manual{MQA,
  author    = {Mengzhao Wang and
               Haotian Wu and
               Xiangyu Ke and 
               Yunjun Gao and
               Xiaoliang Xu and
               Lu Chen},
  title     = {An Interactive Multi-modal Query Answering System with Retrieval-Augmented Large Language Models},
  url       = {xxx},
  year      = {2024}
}

@article{MUST_ICDE2024,
  title={Must: An effective and scalable framework for multimodal search of target modality},
  author={Mengzhao Wang and Xiangyu Ke and Xiaoliang Xu and Lu Chen and Yunjun Gao and Pinpin Huang and Runkai Zhu},
  journal={arXiv preprint arXiv:2312.06397},
  year={2023}
}
```



