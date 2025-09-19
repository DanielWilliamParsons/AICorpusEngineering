#!/bin/zsh

# Resolve the directory where this script lives
# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"        # path to tests_shell
echo $SCRIPT_DIR
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")" # path to LLM
echo $PROJECT_ROOT
LLAMA_BIN="$PROJECT_ROOT/llama.cpp/build/bin/llama-run"
echo $LLAMA_BIN
MODEL="$PROJECT_ROOT/llama-3-8b-instruct.Q4_K_M.gguf"

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

Nowadays, more and more people start to notice the problem of recycling which get a lot of people’s attention. A talk about if we should really do recycling has already appear in many conversation between teachers, parents and also some groups. And in my opinion, even though there are still some troubles, there are many good things people can get from doing recycling. For people themselves, it is an important challenge we should face to protect environment, to sharp our mind of duty in the hard life, and to get more experience in the compare with others who don’t recycle. All of these maybe finally make big help for our future whenever we meet climate problem or when we want a better life. Besides, compared with old people, young people have more free time and more new ideas. There are always some students who spend too much time on computer games, and they can also choose a recycling job which will bring more good influence. And for society, when people join, it will add new power to the recycling system. The energy that people show can push others to be more active, which maybe make the recycle more fast. What’s more, the use of old material is more cheap, so companies can save cost and at the same time get more profit. Since there are so many good points, why not start recycling now? With power and determination, we should take this chance to do it. Don’t wait, just do. Even if some problem stop you, you will get good things for you.
Provide one comment on what was done well and two comments on what could be improved.
Provide a score from 1 to 10 for language, content and organization.

EOF