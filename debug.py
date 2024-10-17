from typing import List, Optional
import fire
from llama import Llama, Dialog

CKPT_dir = "./Llama-2-7b-chat"
TOKENIZER_PATH = "./Llama-2-7b-chat/tokenizer.model"
def main(
    ckpt_dir: str = CKPT_dir,
    tokenizer_path: str = TOKENIZER_PATH,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        # 提示必须按system(可省略)、user、assistant的顺序排列,且最后一轮必须是user
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}]
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"llama: {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n=========================================================================\n")


if __name__ == "__main__":
    fire.Fire(main)