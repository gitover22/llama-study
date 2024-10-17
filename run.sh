#!/bin/bash
$(which torchrun) --nproc_per_node 1 --nnodes 1 example_chat_completion.py \
    --ckpt_dir Llama-2-7b-chat/  \
    --tokenizer_path Llama-2-7b-chat/tokenizer.model