# AICorpusEngineering

This project applies AI engineering approaches to corpus linguistics.

The purpose of this project is to demonstrate the feasibility of using consumer level hardware for corpus linguistics working with large language models and make available AI techniques for doing corpus linguistics for research and language teaching.

The models used in this project are Llama-3-8B-Instruct / Llama-3.1-8B-Instruct series and the GPT-OSS-20B

## Pre-requisites
**Downloads, Specifications, Considerations**

1. Llama-3.1-8B-Instruct series

A good place to access these models is here: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

Different quantizations are available. Quantization refers to how the size of the LLM's parameters were reduced to fit smaller RAM. The model's parameters are originally in 32-bits. Using this model could consume up to 38GB of RAM when inferencing (doing chats) with the model. This isn't really necessary to get good performance. Quantization reduces the size of the bits to save memory. 4-bit models can suffice and still give good performance. A 4-bit model could consume around 9GB of RAM if inferencing a context window (amount of text input and output) of around 4000 tokens. Select the model to download based on your computer's size. Note that the model size downloaded indicates **only** the amount of space the model takes when loaded into memory. When running inferences, more RAM is needed to calculate the next token, and the amount of RAM needed will depend on the context window size. A context window of 4000 tokens could consume an 2 - 4GB of RAM.

Downloading different models and testing with the provided shell tests can indicate which model is best for you.

2. llama.cpp

llama.cpp is a package containing functions that help you run the model, perform diagnositics and even "look inside" the model. This is necessary to work with the model.

First, get the llama.cpp code. Then compile the code. This is necessary to run the model.
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON
cmake --build . --config Release
```

For reference, directories should have the following organization, e.g.:
<pre> 
├──LLM/
|   └── AICorpusEngineering/
|       └── shell_tests/
|           ├── hpt_essay_grading_zero_shot.sh
|           ├── ...
|           └── ...
|       ├── ...
|       └──  README.md
|   ├── llama.cpp/
|       ├── ...
|       └── ...
|   ├── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
|   └── Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf
</pre>

## Run basic tests
There are a few test files written in shell script under the directory shell_tests. In particular, the script `hpt_essay_grading_zero_shot.sh` has been written verbosely so you can check the directories you are in when running the tests.

To run the tests:

1. In the command line navigate to shell_tests directory.
2. Call the test with the model name. The model name assumes the directory is organized as above. If the model is in another folder, adjust the model name accordingly: `./hpt_essay_grading_zero_shot.sh Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf`

The model should load and output a response to a request for grading an essay. While doing this you can inspect memory usage on your computer and adjust the model accordingly.

