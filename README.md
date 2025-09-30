# AICorpusEngineering

This project applies AI engineering approaches to corpus linguistics.

The purpose of this project is to demonstrate the feasibility of using consumer level hardware for corpus linguistics working with large language models and make available AI techniques for doing corpus linguistics for research and language teaching.

The models used in this project are Llama-3-8B-Instruct / Llama-3.1-8B-Instruct series and the GPT-OSS-20B

## Basic Commands from the command line
`pip install -e .`

### Tag texts for parts of speech using SpaCy
Use Parts of Speech tagging from SpaCy to annotate the the syntactic parts of speech of a corpus.

`tag-texts input_dir output_dir --model --workers`
* **input_dir**: The root directory containing your corpus files. Files are annotated recursively.
* **output_dir** The root directory where annotated corpus files will be saved. Directory structure of the original corpus is maintained. Annotated sentences are appended on new lines in each document. Paragraphs are indicated with double new lines.
* **--model** Choose the pos tagging model for SpaCy. Default is en_core_web_trf. This takes the longest to tag the texts, but it is the most accurate. Other models available are 
* **--workers** For parallel processing, choose the number of CPU cores to use based on your machine. Default is set to 2. Using more cores will speed up the process.
It is necessary to download the parts of speech tagging model you wish to use. To use the default, run the following in the command line after your python environment has been set up:

`python3 -m spacy download en_core_web_trf`

To get run other models, `python3 -m spacy download en_core_wen_sm` and `python3 -m spacy download en_core_web_md` and `python3 -m spacy download en_core_web_lg`

### Sample sentences
Sample sentences from a corpus that contain a tag of interest. From the command line:

`process-corpus input_dir results_dir results_file --pos_tag --sample_size`
* **input_dir**: this is the root directory containing your corpus files. Files are extracted recursively.
* **results_dir**: the directory where all results of the sampling will be stored.
* **results_file**: defaults to results.ndjson. Ensure your file is labelled with .ndjson
* **--pos_tag**: Set the parts of speech tag that you are interested in sampling sentences for. Default is _ADV for adverbs.
* **--sample_size**: Set the sample size, i.e., the number of sentences to be sampled. Default is 100.

### Manual Tagger
Loop through all the sampled sentences and manually tag them for semantics

`manual-tagger input_dir file_name`
* **input_dir**: The directory where the sample sentences were stored.
* **file_name**: The name of the file containing the sampled sentences. Must end with .ndjson

The manually tagged sentences will be saved in the same directory as the input directory and will have the same file name as file_name but appended with the label _tagged.

### Semantic annotation of adverbs
Annotate adverbs according to CIRCUMSTANCE, STANCE, FOCUS, LINKING and DISCOURSE. See knowledge base explanation below for details.

`run-adverbs input_dir output_dir --error_logs --data_logs --server_bin --model --server_url`
* **input_dir**: The root directory of the corpus you are interested in tagging.
* **output_dir**: The directory where tagged results will be saved
* **--error_logs**: The location of the error logs that track any errors in outputs from the LLM. Defaults to writing time stamped error log files in the output_dir in ndjson format.
* **--data_logs**: The location where the annotated sentences and related metadata are stored. Defaults to writing time stamped files in the output_dir in ndjson format.
* **--server_bin**: Full path to the LLM server binary, usually located inside llama.cpp folder. Defaults to using an environment variable which can be set with `export LLM_SERVER_BIN=/full/path/to/server/binary`.
* **--model**: LLM model to use. Defaults to an environment variable which can be set with `export LLM_MODEL=/full/path/to/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
* **--server_url**: Location of the server. Default is http://127.0.0.1:8080


### Run an ablation study to annotate adverbs in texts
`run-adverbs-ablation input_dir filename output_dir --error_log --data_logs --server_bin --model --server_url`

* **input_dir**: The directory where the tagged gold standard samples are stored. This is the same as the input_dir used in the manual tagging work and the same as the output directory used in the sampling sentences work. No default set.
* **filename**: The name of the file containing the tagged gold standard samples. This is the same name as the file containing the sample sentences with the label _tagged appended before .ndjson. No default set.
* **output_dir**: The name of the output directory where results of the ablation study will be stored. No default set.
* **--error_logs**: Defaults to the output_dir - recommended not to change.
* **--data_logs**: Defaults to the output_dir - changing renders the output_dir redundant, so recommended not to change
* **--server_bin**: Full path to the LLM server binary, usually located inside llama.cpp folder. Defaults to using an environment variable which can be set with `export LLM_SERVER_BIN=/full/path/to/server/binary`.
* **--model**: LLM model to use. Defaults to an environment variable which can be set with `export LLM_MODEL=/full/path/to/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
* **--server_url**: Location of the server. Default is http://127.0.0.1:8080



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
**SpaCy models for Parts of Speech tagging of texts**

To use SpaCy for parts of speech tagging, download the relevant models. For example:

`python3 -m spacy download en_core_web_trf` will download the transformers tagger - this is the highest quality but slowest tagger.

`python3 -m spacy download en_core_wen_sm` and `python3 -m spacy download en_core_web_md` and `python3 -m spacy download en_core_web_lg` are other alternatives with _sm being the fastest, though least accurate.

**Parts of Speech Tagging**
Once installed, you can now tag corpus texts diretly using the tag_texts command:

`tag-texts Corpus_Texts Corpus_Texts_Tagged --model en_core_web_trf --workers 4`

* The first argument is the path to your input corpus text folder
* The second argument is the path where the tagged texts will be written
* The --model flag selects the SpaCy model (default: en_core_web_trf)
* The --workers flag sets the number of parallel processes (default: 2)

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

A single agent, called syntactic-grouper will determine which group to classify the adverb into the appropriate category and save the data to file.

### Knowledge base
A knowledge base is provided for adverbs (and later for other categories) in the knowledge_base folder.

The agents classes (e.g., BroadGrouperAgents in adverbs_broad_grouper_agents.py) access the knowledge and prepare it for injection into the jinja templates as needed. This allows the knowledge to be stored separately and can be updated independently of the agents.

### Tag files

`run-adverbs pos_tagged_corpus_dir output_dir --data_logs="path/to/datafile.ndjson" --error_logs="path/to/errorfile.ndjson" --server_bin="path/to/server/binary --server_url="url_to_server" --model="path/to/llm.gguf"`

#### --data_logs flag
Your parts of speech tagged corpus is processed one file at a time. For each file, the adverbs in an individual sentence are found, and for each adverb, a query is made to the LLM. The response from the LLM is the data that is output to the data_logs

1. Omit this flag and both a `_data_{timestamp}.ndjson` and `_run_completion_{timestamp}.ndjson` will be created in the output_dir.
2. Pass a directory path: `run-adverbs pos_tagged_corpus_dir output_dir --data_logs=data/` then a new directory called data will be created and `_data_{timestamp}.ndjson` and `_run_completion_{timestamp}.ndjson` will be created in the new data directory.
3. Pass a file path: `run-adverbs pos_tagged_corpus_dir output_dir --data_logs=data/my_data.ndjson` and all data will be saved into `my_data.ndjson` with a `_run_completion_{timestamp}.ndjson` written next to it.
4. Pass just a file name: `run-adverbs pos_tagged_corpus_dir, output_dir, --data_logs=mydata.ndjson` then this file will be created in your current working directory, and completion logs will also be created in your current working directory.

Indicating a file name like in options 3 and 4 will create just a single file in which all data will be appended from one run to another. Options 1 and 2 in which you provide no file name will write new `_data_{timestamp}.ndjson` files each time you start a new run.

### Gold Standard Sentence Sampler
`process-corpus pos_tagged_corpus_dir gold_standard_results_dir results.ndjson`

1. The pos_tagged_corpus_dir refers to the root directory where your corpus text files are located
2. gold_standard_results_dir refers to the location where you will store the results of sampling sentences to build the gold standard
3. results.ndjson: you can change the name; this will collect all sentences containing the word of interest, such as all sentences containing adverbs.