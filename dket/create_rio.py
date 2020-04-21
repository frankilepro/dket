import os
import tempfile

import tensorflow

import dket.data
import tests.utils


def convert_to_indexes(fp):
    indexes = {}
    with open(fp) as f:
        content = f.readlines()
        for c in content:
            indexes[c.strip()] = len(indexes)

    return indexes


def get_index(word, indexes):
    word = clean_word(word)
    if word in indexes:
        return indexes[word]
    return indexes["<UNK>"]


def clean_word(word):
    return word \
        .replace("/<EOS>", "") \
        .replace("/PRP$", "") \
        .replace("/JJR", "") \
        .replace("/JJS", "") \
        .replace("/PDT", "") \
        .replace("/PRP", "") \
        .replace("/RBS", "") \
        .replace("/VDB", "") \
        .replace("/VBD", "") \
        .replace("/VBG", "") \
        .replace("/VBN", "") \
        .replace("/VBP", "") \
        .replace("/VBZ", "") \
        .replace("/WDT", "") \
        .replace("/CB", "") \
        .replace("/CC", "") \
        .replace("/CD", "") \
        .replace("/DT", "") \
        .replace("/FW", "") \
        .replace("/IN", "") \
        .replace("/JJ", "") \
        .replace("/MD", "") \
        .replace("/NN", "") \
        .replace("/RP", "") \
        .replace("/TO", "") \
        .replace("/RB", "") \
        .replace("/VB", "") \
        .replace("/.", "")


def create_rio_base(to_convert, output_file, vocabulary_fp, shortlist_fp, from_curated=False):
    vocabulary_indexes = convert_to_indexes(vocabulary_fp)
    shortlist_indexes = convert_to_indexes(shortlist_fp)
    for i in range(50):
        shortlist_indexes[f"LOC#{i}"] = len(shortlist_indexes)

    with tensorflow.python_io.TFRecordWriter(output_file) as writer:
        for line in to_convert:
            words, formula = tuple(line.split("\t"))
            words = words.split()
            formula = formula.split()
            if from_curated:
                words_indexes = {}
                for c in words:
                    words_indexes[clean_word(c)] = len(words_indexes)
                for i in range(len(formula)):
                    f = formula[i]
                    if f in words_indexes:
                        formula[i] = f"LOC#{words_indexes[f]}"

            words = [get_index(x, vocabulary_indexes) for x in words]
            formula = [get_index(x, shortlist_indexes) for x in formula]
            writer.write(
                dket.data.encode(words, formula)
                .SerializeToString())


def create_rio(to_convert_file, output_file, vocabulary_fp, shortlist_fp, from_curated=False):
    with open(to_convert_file) as file_in:
        content = file_in.readlines()
        content = [x.strip() for x in content]
        create_rio_base(content, output_file, vocabulary_fp, shortlist_fp, from_curated)


if __name__ == "__main__":
    vocabulary_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\vocabulary.idx"
    shortlist_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\shortlist.idx"
    output_file = r"C:\Users\frank\GitHub\dket\datasets\425-curated\validation.rio"
    to_convert_file = r"C:\Users\frank\GitHub\dket\datasets\425-curated\validation.tsv"
    create_rio(to_convert_file, output_file, vocabulary_fp, shortlist_fp, True)
