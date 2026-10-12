from cs336_basics.train_bpe import train_bpe_optimized
from cs336_basics.train_bpe import save_merges, save_vocab

import os
import time
import threading
import psutil

print("Start train_bpe_tinystories.py")

file_path = "data/owt_train.txt"
vocab_size = 32000

time_stamp = time.strftime("%Y%m%d_%H%M")
vocab_path = f"data/owt_vocab_path{time_stamp}.json"
merges_path = f"data/owt_merges_path{time_stamp}.txt"

############# memory monitor def #############
process = psutil.Process(os.getpid())
stop_monitor = threading.Event()
peak_memory_bytes = 0

def monitor_memory():
    global peak_memory_bytes
    while not stop_monitor.is_set():
        total_rss = 0
        try:
            total_rss += process.memory_info().rss
            children = process.children(recursive=True)
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except(psutil.NoSuchProcess, psutil.AccessDenied,):
                    pass
            peak_memory_bytes = max(peak_memory_bytes, total_rss)
        except(psutil.NoSuchProcess, psutil.AccessDenied,):
            pass
        time.sleep(0.05)

monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
############# memory monitor def #############

monitor_thread.start()

start = time.perf_counter()

try:
    vocab, merges = train_bpe_optimized(input_path=file_path, 
                                        vocab_size=vocab_size, 
                                        special_tokens=["<|endoftext|>"])
    
finally:
    end = time.perf_counter()
    stop_monitor.set()
    monitor_thread.join()

training_time = end - start

peak_memory_bytes = peak_memory_bytes / 1024**2

longest_token = max(vocab.values(), key=len)
longest_token_length = len(longest_token)

save_vocab(vocab=vocab, path=vocab_path)
save_merges(merges=merges, path=merges_path)

print(f"Saved vocabulary in file {vocab_path}. ")
print(f"Saved merges in file {merges_path}. ")
print(f"Training time: {training_time:.2f} seconds.")
print(f"Peak memory (main + workers) is {peak_memory_bytes:.2f} MB.")
print(f"Longest token: {longest_token!r}") # 调试 str、bytes、token 这类数据时，!r 很常用，相当于repr()。
print(f"Longest token length: {longest_token_length} bytes.")

try:
    print(f"Decoded longest token: {longest_token.decode('utf-8')!r}")
except UnicodeDecodeError:
    print("Decoded longest token: not valid UTF-8.")