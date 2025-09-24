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
|       └── src
|           └──AICorpusEngineering
|               └── shell_tests/
|                   ├── hpt_essay_grading_zero_shot.sh
|                   ├── ...
|                   └── ...
|               ├── ...
|   └──  README.md
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

## Preparing to run Python Scripts
First install the package in editable mode

`pip install -e .`

This makes the command-line entry points available globally in your environment.

## Preparing texts for parts of speech tagging
**SpaCy models for tagging text**

To use SpaCy for parts of speech tagging, download the relevant models. For example:

`python3 -m spacy download en_core_web_trf` will download the transformers tagger - this is the highest quality but slowest tagger.

`python3 -m spacy download en_core_wen_sm` and `python3 -m spacy download en_core_web_md` and `python3 -m spacy download en_core_web_lg` are other alternatives with _sm being the fastest, though least accurate.

**Tagging**
Once installed, you can now tag corpus texts diretly using the tag_texts command:

`tag_texts Corpus_Texts Corpus_Texts_Tagged --model en_core_web_trf --workers 4`

* The first argument is the path to your input corpus text folder
* The second argument is the path where the tagged texts will be written
* The --model argument selects the SpaCy model (default: en_core_web_trf)
* The --workers argument sets the number of parallel processes (default: 2)

The directory structure might look something like this:

<pre> 
├──LLM/
|   └── AICorpusEngineering/
|       └── src
|           └──AICorpusEngineering
|               └── shell_tests/
|                   ├── hpt_essay_grading_zero_shot.sh
|                   ├── ...
|                   └── ...
|               ├── ...
|   └──  README.md
|   ├── llama.cpp/
|       ├── ...
|       └── ...
|   ├── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
|   ├── Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf
|   └── Corpus_Texts/
|       └── text_set1/
|           ├── text1.txt
|           ├── text2.txt
|           ├── text3.txt
|           ├── ...
|       └── text_set2
|           ├── ...
|           └── ...
|   └── Corpus_Texts_Tagged/
|       └── text_set1/
|           ├── text1.txt
|           ├── text2.txt
|           ├── text3.txt
|           ├── ...
|       └── text_set2
|           ├── ...
|           └── ...
</pre>

## Broad Design of the LLM-based Tagging System
The system uses object-oriented patterns to retrieve data, pass data to and from the LLM agents and to keep records of the data.

### Adverbs Broad Grouping Agents
Three agents are designed to identify and verify the broad category to which a selected adverb belongs:
1. Circumstantial adverb (time, place, manner, etc)
2. Stance adverb (attitude, evaluation, etc)
3. Linking adverb (addition, result, etc.)
4. Discourse adverb (manage interaction, manage flow of discourse)

The three agents are:
1. A broad grouping agent: uses knowledge-base + few-shot chain-of-thought prompting to identify the category
2. A verifcation agent: uses knowledge-base + chain-of-thought prompting to verify the decision of the first agent
3. A voting agent: uses knowledge-base + generated knowledge and chain-of-thought prompting to mediate the first two agents if there is disagreement and reach a final decision

### Knowledge base
A knowledge base is provided for adverbs (and later for other categories) in the knowledge_base folder.

The agents classes (e.g., BroadGrouperAgents in adverbs_broad_grouper_agents.py) access the knowledge and prepare it for injection into the jinja templates as needed. This allows the knowledge to be stored separately and can be updated independently of the agents.

### Tag files

`run-adverbs pos_tagged_corpus_dir output_dir --data_logs `

#### --data_logs flag
Your parts of speech tagged corpus is processed one file at a time. For each file, the adverbs in an individual sentence are found, and for each adverb, a query is made to the LLM. The response from the LLM is the data that is output to the data_logs

1. Omit this flag and both a `_data_{timestamp}.ndjson` and `_run_completion_{timestamp}.ndjson` will be created in the output_dir.
2. Pass a directory path: `run-adverbs pos_tagged_corpus_dir output_dir --data_logs=data/` then a new directory called data will be created and `_data_{timestamp}.ndjson` and `_run_completion_{timestamp}.ndjson` will be created in the new data directory.
3. Pass a file path: `run-adverbs pos_tagged_corpus_dir output_dir --data_logs=data/my_data.ndjson` and all data will be saved into `my_data.ndjson` with a `_run_completion_{timestamp}.ndjson` written next to it.
4. Pass just a file name: `run-adverbs pos_tagged_corpus_dir, output_dir, --data_logs=mydata.ndjson` then this file will be created in your current working directory, and completion logs will also be created in your current working directory.

Indicating a file name like in options 3 and 4 will create just a single file in which all data will be appended from one run to another. Options 1 and 2 in which you provide no file name will write new `_data_{timestamp}.ndjson` files each time you start a new run.


