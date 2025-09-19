#!/bin/zsh

# Resolve the directory where this script lives
# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # path to tests_shell
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")" # path to LLM
LLAMA_BIN="$PROJECT_ROOT/llama.cpp/build/bin/llama-run"
MODEL_DIR="$PROJECT_ROOT"

# Allow model file to be passed in as an argument, otherwise use default
if [ -n "$1" ]; then
  MODEL="$MODEL_DIR/$1"
else
  MODEL="$MODEL_DIR/Meta-Llama-3.1-8b-Instruct-Q4_K_M.gguf"
fi

if [ ! -f "$LLAMA_BIN" ]; then
  echo "❌ llama-run not found at $LLAMA_BIN"
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "❌ model not found at $MODEL"
  exit 1
fi

"$LLAMA_BIN" \
  --context-size 8192 \
  --temp 0.6 \
  --threads 6 \
  "$MODEL" << EOF
  Read the following second language learner essay:
All of people know about poverty in the world. Of course, I think this topic may depend on countries situation. From my point of view, there are many reasons that poverty is serious problem, especially "Why poverty is important to talk about?" The first reason that is important because some families do not have enough money to buy food and clothes. So the children cannot go to school or have good health. The second reason is, whenever people are sick, but they would not like to disturb other people, they cannot pay for hospital by themselves. The third reason, if the people would like to use their time effectively, they need job, but many poor people do not have chance to get work. Beside they waste their time without job, they also feel stress with their life. The fourth reason, poverty makes people being really sad and lose hope, because they cannot change their situation easy. For the last reason, poverty can be a thing that makes countries have many problems, such as crime, corruption and lack of education. To sum up, I would not like to focus on "poverty is important or unimportant for people," but if we talk about poverty, we can learn the true meaning of helping and supporting each other.

Comment on strengths
One positive thing is the essay try to explain poverty with many reasons. The writer give five different points, like no money for food and school, cannot pay hospital, no chance for job, feeling sad and lose hope, and big problems for countries. This shows the writer can develop the topic and keep it connect to the question.
Two comments on improvements
Language:
The essay has many grammar mistakes, so sometimes is not easy to understand. For example, "countries situation" and "cannot change their situation easy." The writer use same words many times, like "money" and "job." The essay will be better if use more correct forms and some different vocabulary.

Organization:
The essay use "first reason, second reason…" but the sentences are too long and sometimes mix ideas. The start is too general and not clear opinion, and the ending repeat ideas but also little confusing. The essay can improve if the writer make clear sentences, better connection, and strong last sentence.
Language 4/10:
Many grammar errors, sometimes make sentence hard to read, but meaning is usually clear. Vocabulary is simple and often repeat ("money," "job"), and some words not correct ("countries situation," "change their situation easy").

Content 7/10:
The essay give five reasons about poverty, like no food, no hospital, no job, sad feeling, and country problems. The ideas are short and not explain deep, but still clear and connect to the topic.

Organization 7/10:
The writer use "first reason, second reason…" to order ideas, but sentences are too long and sometimes mix ideas. The introduction is very general and the conclusion is weak because it confuse the main point. The sequence is okay, but transition and flow are limited.
Final Score Summary

Language: 4/10
Content: 7/10
Organization: 7/10
Overall: 18/30

  Read the following second language learner essay:
Nowadays, more and more people start to notice the problem of recycling which get a lot of people's attention. A talk about if we should really do recycling has already appear in many conversation between teachers, parents and also some groups. And in my opinion, even though there are still some troubles, there are many good things people can get from doing recycling. For people themselves, it is an important challenge we should face to protect environment, to sharp our mind of duty in the hard life, and to get more experience in the compare with others who don’t recycle. All of these maybe finally make big help for our future whenever we meet climate problem or when we want a better life. Besides, compared with old people, young people have more free time and more new ideas. There are always some students who spend too much time on computer games, and they can also choose a recycling job which will bring more good influence. And for society, when people join, it will add new power to the recycling system. The energy that people show can push others to be more active, which maybe make the recycle more fast. What’s more, the use of old material is more cheap, so companies can save cost and at the same time get more profit. Since there are so many good points, why not start recycling now? With power and determination, we should take this chance to do it. Don’t wait, just do. Even if some problem stop you, you will get good things for you.
  Provide one comment on what was done well and two comments on what could be improved.
  Provide a score from 1 to 10 for language, content and organization.
Let's think step by step
EOF