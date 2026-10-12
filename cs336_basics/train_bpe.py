# bpe(byte-pair-encoding) implementation
# 
# 【0】 multiprocessing optimization - chunk the input text into smaller pieces and process them in parallel
# ( e.g., "doc A<|endoftext|>doc B" can be split before "<|endoftext|>" ), but not inside a document or inside a special token.
#
# in each parallel process (inside each chunk, e.g, doc A and doc B), we will do:
# 【1】 regex-based pretokenization (i.e., split text into pre-tokens) according to the special characters
# e.g., "some text that i'll pre-tokenize" → ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
# 【2】 counting the frequency of each pair of bytes or merged tokens in the pre-tokens, and weighting the counts by the number of pre-tokens in the chunk.
# the step 1 and step 2 are actually done using "re.finditer" to get the pre-tokens and their counts in one pass, instead of doing them separately.
#
# from all parallel processes, accumulate the counts and then:
# 【3】 merging the most frequent pair of bytes into a new bytes token, and updating the frequency counts of the pairs that include the new token.
# 【4】 repeating steps 2 and 3 until vocab_size is reached, or no more pairs can be merged.
#
# the data streams follows: 
# 【bytes file → decoded text → regex-based pre-tokens (encoded byte sequences) 
#   → {pre-tokens: frequency} → {encoded byte str: frequency} → merged byte str vocab】
#
# bytes file
# → decoded text: str
# → 先按 special tokens 分段
# → 每一段做 regex，得到 pre-token 字符串
# → 统计每种 pre-token 的频率
# → 每种 pre-token 转成 UTF-8 byte 序列
# → 统计 pair，开始全局 merge

import regex as re
import os
from typing import BinaryIO
from multiprocessing import Pool
import json
from tqdm import tqdm


def find_chunk_boundaries(file: BinaryIO, 
                          split_special_token: bytes
                          ) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    desired_num_chunks = os.cpu_count() or 1
    chunk_size = file_size // desired_num_chunks
        
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
        
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
        
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
        
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
        
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
        
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def bpe_worker(input_path: str, 
               start: int, 
               end: int, 
               special_tokens: list[str]
               ) -> dict[bytes, int]:
    """
    Worker function to process a chunk of the file and return the frequency counts of pre-tokens in that chunk.
    """
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")
    if special_tokens is None or special_tokens == []:
        segments = [text] 
    else:
        segments = re.split("|".join(re.escape(special_token) for special_token in special_tokens), text)

    counts = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for seg in segments:
        for match in re.finditer(PAT, seg):
            pre_token = match.group()
            counts[pre_token.encode("utf-8")] = counts.get(pre_token.encode("utf-8"), 0) + 1 # the pre-token representation in bytes, e.g., b"some"
        
    return counts


def train_bpe(input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]
        )-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the input text.
    """
    # 0. initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8") # the token representation in bytes, e.g., b"<|endoftext|>"
    merges = []

    # 1. regex-based pretokenization using multiprocessing
    file = open(input_path, "rb")
    chunk_boundaries = find_chunk_boundaries(file, special_tokens[0].encode("utf-8"))
    start_end_pairs = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))

    with Pool() as pool:
        results = pool.starmap(bpe_worker, [(input_path, start, end, special_tokens) for start, end in start_end_pairs])

    # the current token sequences in the vocabulary, represented as lists of bytes
    # (fixed) token_list 是 list，不能作为字典 key；PDF 才建议用 tuple[bytes, ...]，因为 tuple 可哈希，最后要进行一个转换

    pretoken_counts = {}
    for chunk_counts in results:
        for pretoken, count in chunk_counts.items(): # accumulate from all chunks
            pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + count

    current_token_sequence = {}
    for pretoken, count in pretoken_counts.items():
        token_tuple = tuple([pretoken[i:i+1] for i in range(len(pretoken))]) # if pre_token[i], we obtain the integer value of the byte.
        current_token_sequence[token_tuple] = current_token_sequence.get(token_tuple, 0) + count

    num_merges = vocab_size - len(vocab) # for tqdm

    # 2. counting the pair-frequency and 3. repeatedly merges and frequency updates
    with tqdm(total=num_merges, desc="Training BPE") as pbar:
        while len(vocab) < vocab_size:
            pair_counts = {}
            for token_tuple, count in current_token_sequence.items():
                for i in range(len(token_tuple) - 1):
                    pair = (token_tuple[i], token_tuple[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count

            if not pair_counts:
                break

            # merge
            most_frequent_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair)) # (fixed) 实现“频率并列时取字典序更大的 pair”
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            vocab[len(vocab)] = new_token
            merges.append(most_frequent_pair)
         
            # update the current_token_sequence with the new merged token
            next_token_sequence = {}
            for token_tuple in list(current_token_sequence.keys()):
            
                temp_list = []
                j = 0
                while j < len(token_tuple):
                    if j + 1 < len(token_tuple) and (token_tuple[j], token_tuple[j + 1]) == most_frequent_pair:
                        temp_list.append(new_token)
                        j += 2
                    else:
                        temp_list.append(token_tuple[j])
                        j += 1
                new_token_tuple = tuple(temp_list)
                next_token_sequence[new_token_tuple] = next_token_sequence.get(new_token_tuple, 0) + current_token_sequence[token_tuple]

            current_token_sequence = next_token_sequence # continue the while loop

            pbar.update(1)

    file.close()

    return vocab, merges


def save_vocab(vocab, path):
    """save the vocab dict[int, bytes] into json file."""
    # bytes cannot be serialized by JSON directly, so convert them to hex strings
    serializable_vocab = {token_id: token_bytes.hex() 
                          for token_id, token_bytes in vocab.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f)  # the token_id will be in str form

def save_merges(merges, path):
    """save the merges list[tuple[bytes, bytes]] into txt file."""
    with open(path, "w", encoding="utf-8") as f:
        for token1, token2 in merges:
            f.write(f"{token1.hex()}\t{token2.hex()}\n")

def load_vocab(vocab_path):
    """load the vocab dict[int, bytes] from json file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = {int(token_id): bytes.fromhex(token_bytes) 
             for token_id, token_bytes in data.items()}
    return vocab

def load_merges(merges_path):
    """load the merges list[tuple[bytes, bytes]] from txt file."""
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            token1, token2 = line.rstrip("\n").split("\t")
            merges.append((bytes.fromhex(token1),
                          bytes.fromhex(token2)))
    return merges


########################## BPE optimization ##########################
import heapq
from collections import Counter


def train_bpe_optimized(input_path: str, 
                        vocab_size: int,
                        special_tokens: list[str]
                        )-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the input text.
    """
    # pre-tokenization
    # ↓
    # seq_id -> mutable sequence
    # seq_id -> count
    # ↓
    # 初始化：
    # pair_freq
    # pair_to_seq_ids
    # heap
    # ↓
    # while:
    #   heap 取 most_frequent_pair
    #   ↓
    #   pair_to_seq_ids 找受影响 sequences
    #   ↓
    #   只扫描这些 sequences 做 merge
    #   ↓
    #   根据 old pairs / new pairs 增量更新 pair_freq
    #   ↓
    #   更新 pair_to_seq_ids
    #   ↓
    #   新 frequency push 回 heap

    # vocab initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8") # the token representation in bytes, e.g., b"<|endoftext|>"
    merges = []

    # multiprocessing pretokenization
    file = open(input_path, "rb")
    chunk_boundaries = find_chunk_boundaries(file, special_tokens[0].encode("utf-8"))
    start_end_pairs = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))

    with Pool() as pool:
        results = pool.starmap(bpe_worker, [(input_path, start, end, special_tokens) for start, end in start_end_pairs])

    pretoken_counts = {}
    for chunk_result in results:
        for sequence, count in chunk_result.items():
            pretoken_counts[sequence] = pretoken_counts.get(sequence, 0) + count

    token_seq = {}
    seq_count = {}
    for seq_id, seq in enumerate(pretoken_counts):
            token_seq[seq_id] = [seq[i:i+1] for i in range(len(seq))]  # seq_id -> mutable sequence
            seq_count[seq_id] = pretoken_counts[seq]  # seq_id -> count

    # initialize pair_freq, pair_to_seq_id, seq_pair_counts
    pair_freq = {}
    pair_to_seq_ids = {}
    seq_pair_counts = {}
    for seq_id, seq in token_seq.items():

        count = seq_count[seq_id]
        local_counts = {}

        for j in range(len(seq) - 1):
            pair = (seq[j], seq[j+1])
            local_counts[pair] = local_counts.get(pair, 0) + 1
            
        seq_pair_counts[seq_id] = local_counts

        for pair, local_count in local_counts.items():
            pair_freq[pair] = pair_freq.get(pair, 0) + local_count * count
            pair_to_seq_ids.setdefault(pair, set()).add(seq_id)

    # initialize heap for seaching max_freq_pair
    heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)

    num_merges = vocab_size - len(vocab) # for tqdm
    with tqdm(total=num_merges, desc="Training BPE") as pbar:
        while len(vocab) < vocab_size:

            most_freq_pair = None

            # heap + lazy update -> get the most_freq_pair
            while heap:
                neg_freq, pair = heapq.heappop(heap)
                freq = -neg_freq
                if pair_freq.get(pair, 0) != freq: #lazy update
                    continue

                candidates = {pair}
                while heap and -heap[0][0] == freq: # get all same freq pair into candidates
                    neg_freq_same, pair_same = heapq.heappop(heap)
                    if pair_freq.get(pair_same, 0) != freq: #lazy update
                        continue
                    candidates.add(pair_same)

                most_freq_pair = max(candidates)

                for other_pair in candidates:
                    if other_pair != most_freq_pair:
                        heapq.heappush(heap, (-freq, other_pair))
                break

            if most_freq_pair is None:
                break
            

            # merge
            new_token = most_freq_pair[0] + most_freq_pair[1]
            vocab[len(vocab)] = new_token
            merges.append(most_freq_pair)

            # get affected_seq_ids (a list, might have a lot)
            affected_seq_ids = list(pair_to_seq_ids.get(most_freq_pair, set()))

            changed_pairs = set()
            # merge affected sequences and update locally
            for seq_id in affected_seq_ids:

                seq = token_seq[seq_id]
                count = seq_count[seq_id]
                local_counts = seq_pair_counts[seq_id]

                ######## update_pair func start ########
                def update_pair(pair, delta):

                    old_local_count = local_counts.get(pair, 0)
                    new_local_count = old_local_count + delta

                    if new_local_count > 0:
                        local_counts[pair] = new_local_count
                    else:
                        local_counts.pop(pair, None)

                    pair_freq[pair] = (pair_freq.get(pair, 0) + delta * count)
                    changed_pairs.add(pair)

                    if old_local_count == 0 and new_local_count > 0:
                        pair_to_seq_ids.setdefault(pair, set()).add(seq_id)

                    elif old_local_count > 0 and new_local_count == 0:
                        seq_ids = pair_to_seq_ids.get(pair)
                        if seq_ids is not None:
                            seq_ids.discard(seq_id)
                            if not seq_ids:
                                pair_to_seq_ids.pop(pair, None)
                ######## update_pair func end ########

                j = 0

                while j < len(seq) - 1:

                    if (seq[j], seq[j+1]) != most_freq_pair:
                        j += 1
                        continue

                    left = seq[j-1] if j > 0 else None
                    right = seq[j+2] if j+2 < len(seq) else None
                    a = seq[j]
                    b = seq[j+1]

                    if left is not None:
                        update_pair((left, a), -1)
                    update_pair((a, b), -1)
                    if right is not None:
                        update_pair((b, right), -1)

                    seq[j] = new_token
                    del seq[j+1]

                    if left is not None:
                        update_pair((left, new_token), 1)
                    if right is not None:
                        update_pair((new_token, right), 1)

                    j += 1

            for pair in changed_pairs:
                freq = pair_freq.get(pair, 0)
                if freq > 0:
                    heapq.heappush(heap, (-freq, pair))
                else:
                    pair_freq.pop(pair, None)

            pbar.update(1)

    file.close()
    
    return vocab, merges
