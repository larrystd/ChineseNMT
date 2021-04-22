'''
Author: your name
Date: 2021-04-22 09:05:58
LastEditTime: 2021-04-22 09:53:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ChineseNMT/utils.py
'''
import os
import logging
import sentencepiece as spm

"""
sentencepiece工具，
https://github.com/google/sentencepiece

提供四种关于词的切分方法。这里跟中文的分词作用是一样的，但从思路上还是有区分的。通过使用我感觉：在中文上，就是把经常在一起出现的字组合成一个词语；在英文上，它会把英语单词切分更小的语义单元，减少词表的数量。

sentencpiece更倾向输出更大粒度的词，像把“机器学习领域中”放在一起，说明这个词语在语料库中出现的频率很高。

BPE(Byte Pair Encoding)最早是一种压缩算法，基本思路是将使用最频繁的字节用一个新的字节组合代替，比如用字符的n-gram替换各个字符。例如，假设('A', 'B') 经常顺序出现，
则用一个新的标志'AB'来代替它们。
"""
def chinese_tokenizer_load():
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("./tokenizer/chn"))
    return sp_chn


def english_tokenizer_load():
    """导入词表"""
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('{}.model'.format("./tokenizer/eng"))
    return sp_eng


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


