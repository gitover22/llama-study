# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List
# 使用 SentencePiece 分词器
from sentencepiece import SentencePieceProcessor


logger = getLogger()


class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""
    def __init__(self, model_path: str):
        """
        用SentencePiece model初始化Tokenizer
        

        Args:
            model_path (str): The path to the SentencePiece model file.   ---->   ('./Llama-2-7b-chat/tokenizer.model')
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id() # 开始BOS
        self.eos_id: int = self.sp_model.eos_id() # 结束EOS
        self.pad_id: int = self.sp_model.pad_id() # pad填充标记
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size() # self.sp_model.vocab_size()词汇表大小为32000

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)
