from collections.abc import Iterable, Iterator
from cs336_basics.train_bpe import load_merges, load_vocab
import regex as re

class bpe_Tokenizer:
    def __init__(self, 
                 vocab, 
                 merges, 
                 special_tokens=None,
                 ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. 
        This function should accept the following parameters:
        vocab: dict[int, bytes]  
        merges: list[tuple[bytes, bytes]]  
        special_tokens: list[str] | None = None  
        """
        self.vocab = vocab # ID → bytes
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte2id = {token_bytes: token_int 
                        for token_int, token_bytes in vocab.items()}
        self.pair2rank = {pair: rank 
                          for rank, pair in enumerate(merges)}
         
    @classmethod
    def from_files(cls, 
                   vocab_filepath, 
                   merges_filepath, 
                   special_tokens=None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
        same format that your BPE training code output) and (optionally) a list of special tokens. 
        This method should accept the following additional parameters:
        vocab_filepath: str  
        merges_filepath: str  
        special_tokens: list[str] | None = None  
        """
        vocab = load_vocab(vocab_path=vocab_filepath)
        merges = load_merges(merges_path=merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, 
               text: str
               ) -> list[int]: 
        """
        Encode an input text into a sequence of token IDs.
        text → special token sectioned → regex pre-token → UTF-8 bytes 
        → merge using trained merges → token IDs.
        """
        # 普通 re.split 会把分隔符从结果中删除。但 tokenizer encode 时：
        # "hello<|endoftext|>world"
        # 不能只编码 hello 和 world；中间的 <|endoftext|> 也必须作为一个完整 token，查到它自己的 ID 并插入输出序列。
        byte_2_id = self.byte2id
        special_tokens = self.special_tokens
        pair2rank = self.pair2rank

        if special_tokens is None or special_tokens == []:
            segments = [text]
        else:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True) # 长token需要有机会作为整体匹配
            escaped = "|".join(re.escape(special_token) for special_token in sorted_special_tokens)
            pattern = f"({escaped})"
            segments = re.split(pattern, text) # 多一层括号可以把special_tokens捕获进结果里

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_id_seq = []

        for seg in segments:
            if special_tokens is not None and seg in special_tokens:
                token_id = byte_2_id[seg.encode("utf-8")]
                token_id_seq.append(token_id)
            else:
                for match in re.finditer(PAT, seg):
                    pre_token = match.group().encode("utf-8")
                    token_seq = [pre_token[i:i+1] for i in range(len(pre_token))]
                    while True:
                        to_be_merged = []
                        for i in range(len(token_seq) - 1):
                            pair = (token_seq[i], token_seq[i+1])
                            if pair in pair2rank:
                                to_be_merged.append(pair)
                        if not to_be_merged:
                            break
                        pair_to_merge = min(to_be_merged, key=lambda pair: pair2rank[pair])
                        new_token_seq = []
                        j = 0
                        while j < len(token_seq):
                            if j + 1 < len(token_seq) and (token_seq[j], token_seq[j+1]) == pair_to_merge:
                                new_token_seq.append(token_seq[j]+token_seq[j+1])
                                j += 2
                            else:
                                new_token_seq.append(token_seq[j])
                                j += 1
                        token_seq = new_token_seq
                    for token in token_seq:
                        token_id_seq.append(byte_2_id[token])
        return token_id_seq
    
    def encode_iterable(self, 
                        iterable: Iterable[str]
                        ) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, 
               ids: list[int]
               ) -> str: 
        """
        Decode a sequence of token IDs into text.
        token IDs → token bytes → concatenate bytes → UTF-8 text.
        """
        all_token_bytes = []
        for token_id in ids:
            all_token_bytes.append(self.vocab[token_id])
        text_bytes = b"".join(all_token_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text