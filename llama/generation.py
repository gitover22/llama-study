# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

# # 三种角色
# system 提供框架或者模型前缀
# user 发起对话的用户，提出问题或任务；
# assistant 模型：根据用户的需求和系统的指引生成合适的响应。
Role = Literal["system", "user", "assistant"]

# Message结构体用于存储提示（prompt）
class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            # os.environ["MASTER_ADDR"] = "localhost"
            # os.environ["MASTER_PORT"] = "12355"
            # rank = int(os.environ.get("RANK", 0))  # 全局进程排名
            # # world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))  # 总进程数
            # world_size = int(os.environ.get("WORLD_SIZE", 1))  # 总进程数
            # torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
            # torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id # -1
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos) # logits.shape: [bz, seq_len, vocab_size]
            if temperature > 0:
                # 设置了温度值会使用top-p采样
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1) # logits[:, -1].shape:[bz,vocab_size] 最后一个token的logits
                next_token = sample_top_p(probs, top_p)
            else:
                # 直接取概率最大的
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            # 判断生成的token是否为结束符(eos)，并更新eos_reached标志
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            # 更新上一个位置的指针
            prev_pos = cur_pos
            
            # 如果所有样本都已到达结束符(eos)，则终止生成
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []

        # 遍历每个批次的tokens，将其转换为Python的列表进行处理
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            # 根据是否需要回显(prompt的内容是否包含在生成内容中)，确定截取的起始位置
            # 如果需要回显(echo=True)，则从第0个token开始；否则从提示词之后的token开始
            start = 0 if echo else len(prompt_tokens[i])
            # 将生成的tokens裁剪到提示词长度加上最大生成长度的范围内
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # 如果生成的tokens中包含结束符(eos)，则将tokens裁剪到结束符位置之前
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            # 将处理后的tokens和logprobs分别添加到输出列表中
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        使用语言生成模型为会话对话列表生成响应.

        参数:
        dialogs (List[Dialog]): 对话列表，其中每个对话是消息的列表。
        temperature (float, 可选): 用于控制采样随机性的温度值。默认值为 0.6。
        top_p (float, 可选): 核心采样的 top-p 概率阈值。默认值为 0.9。
        max_gen_len (Optional[int], 可选): 生成的响应序列的最大长度。如果未提供，则设置为模型最大序列长度减 1。
        logprobs (bool, 可选): 是否计算生成的每个词元的对数概率的标志。默认值为 False。
        返回:
        List[ChatPrediction]: 包含生成的响应的预测列表。
        抛出:
        AssertionError: 如果对话中最后一条消息不是来自用户。
        AssertionError: 如果对话中的角色顺序不是按要求的“用户”、“助手”和可选的“系统”顺序。
        注意:
        此方法为提供的对话生成助手的响应。
        它采用核心采样 (nucleus sampling) 来在文本生成中引入可控的随机性。
        如果 logprobs 为 True, 将为每个生成的词元计算对数概率。

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            # 检查对话中的消息是否包含特殊标签（如 [INST] 或 <<SYS>> 等），这些标签不允许出现在提示中。
            # 如果包含，则将该对话标记为不安全请求（unsafe_requests）。
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )

            # 如果对话的第一条消息的角色是 'system'，则将该系统消息内容添加到对话的第二条消息（通常是用户消息）中，
            # 并且保留其他的后续消息。
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],  # 将第二条消息的角色（通常是用户）赋值给新创建的消息
                        "content": B_SYS  # 在对话内容的开头添加系统标签 <<SYS>>
                        + dialog[0]["content"]  # 插入系统消息内容
                        + E_SYS  # 在对话内容的末尾添加系统结束标签 <</SYS>>
                        + dialog[1]["content"],  # 加上第二条消息（通常是用户消息）的内容
                    }
                ] + dialog[2:]  # 保留从第三条消息开始的所有对话内容

            # 确认对话中偶数索引（0, 2, 4...）的消息角色都是 'user'，奇数索引（1, 3, 5...）的消息角色都是 'assistant'。
            # 如果角色顺序不符合要求，则会触发断言错误。
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"  # 错误信息，提示必须按系统、用户、助手的顺序排列。
            )

            # 将对话中的消息进行编码，将用户的提示和助手的回复串联起来，并为每个对话生成相应的 token 列表。
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",  # 用 [INST] 和 [/INST] 包围消息内容
                        bos=True,  # 在每条提示开头加上开始标志（BOS）
                        eos=True,  # 在每条消息末尾加上结束标志（EOS）
                    )
                    for prompt, answer in zip(  # 每次遍历一对提示和回复（用户提示和助手回复）
                        dialog[::2],  # 用户的消息（偶数索引）
                        dialog[1::2],  # 助手的回复（奇数索引）
                    )
                ],
                [],  # 将所有编码结果拼接成一个完整的 token 列表
            )

            # 确保对话的最后一条消息必须是来自用户的。如果不是，则抛出断言错误。
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"

            # 编码最后一个用户消息，并将其追加到 token 列表中，但不加结束标志（EOS）。
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )

            # 将生成的 token 列表添加到 prompt_tokens 列表中，以便后续生成模型使用。
            prompt_tokens.append(dialog_tokens) # 注意prompt_tokens是一个二维列表

        # 调用生成函数，使用给定的 prompt_tokens 生成响应。
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )

        # 如果需要生成的 token 对应的 log 概率，则返回包含生成文本、token 和 log 概率的预测结果。
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",  # 助手生成的回复角色为 'assistant'
                        "content": self.tokenizer.decode(t)  # 将生成的 token 解码为文本
                        if not unsafe  # 如果对话不包含不安全的特殊标签，则返回生成的文本
                        else UNSAFE_ERROR,  # 否则返回错误信息
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],  # 生成的每个 token 对应的文本
                    "logprobs": logprobs_i,  # 每个生成的 token 对应的 log 概率
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests  # 对生成的 tokens、log 概率和安全性进行迭代
                )
            ]

        # 如果不需要 log 概率，则只返回生成的文本。
        return [
            {
                "generation": {
                    "role": "assistant",  # 助手生成的回复角色为 'assistant'
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,  # 根据安全性返回生成的文本或错误信息
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)  # 对生成的 tokens 和安全性进行迭代
        ]


# top-p采样策略的实现
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
