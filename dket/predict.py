from dket import analytics
# from dket import runtime
# from dket import create_rio
import os
import tempfile


def predict(input_sentence, CONFIG, config, ARGS):
    # the rake is a hand tool that is used by gardener . <EOS>
    # input_sentence = r"a direct ossification is a ossification that do not require the replacement of pre-existing tissue . <EOS>"
    # input_sentence = input_sentence.strip().split()
    input_sentence = input_sentence.strip().split()
    dummy_list = ["0" for i in range(len(input_sentence))]
    input_sentence.append("\t")
    input_sentence.extend(dummy_list)
    input_sentence = " ".join(input_sentence).replace(" \t ", "\t")

    tmp_dir = tempfile.mkdtemp()
    tmp_file = os.path.join(tmp_dir, "eval.rio")

    basedir = os.path.dirname(CONFIG)
    datadir = os.path.dirname(config["train.files"])
    vocabulary_fp = os.path.abspath(os.path.join(basedir, datadir, "vocabulary.idx"))
    shortlist_fp = os.path.abspath(os.path.join(basedir, datadir, "shortlist.idx"))
    output_file = tmp_file
    create_rio.create_rio_base([input_sentence], output_file, vocabulary_fp, shortlist_fp)

    config["eval.files"] = tmp_file
    experiment = runtime.Experiment.load(CONFIG, config, logdir=ARGS.logdir, force=ARGS.force)
    experiment.predict(ARGS.last_checkpoint)

    vocabulary = analytics.load_vocabulary(vocabulary_fp)
    shortlist = analytics.load_vocabulary(shortlist_fp)
    with open(os.path.join(experiment._logdir, "eval", "dump", "dump-20001.tsv")) as the_file:
        for line in the_file:
            input_sentence = line
    prediction = analytics.convert(input_sentence, 0, vocabulary, shortlist)
    print(f"Prediction: {prediction['example']['prediction']}")


def convert_source_to_indexes(source):
    indexes = {}
    for c in source:
        indexes[c.strip()] = len(indexes)

    return indexes


def convert_to_indexes(fp):
    indexes = {}
    with open(fp) as f:
        content = f.readlines()
        for c in content:
            indexes[c.strip()] = len(indexes)

    return indexes


def pad_with_zeros(arr):
    for i in range(50 - len(arr)):
        arr.append(0)


def coonvert_dket_to_embeddings(vocabulary_fp, shortlist_fp, dket_path, dket_emb_path):
    vocabulary_indexes = convert_to_indexes(vocabulary_fp)
    shortlist_indexes = convert_to_indexes(shortlist_fp)
    with open(dket_path) as dket_file, open(dket_emb_path, "w") as dket_emb_file:
        for line in dket_file:
            source, target, prediction = line.strip().split("\t")
            source = source.split()
            target = target.split()
            prediction = prediction.split()

            source_indexes = convert_source_to_indexes(source)
            source = [vocabulary_indexes[x] for x in source]
            target = [shortlist_indexes[x] if x in shortlist_indexes
                      else (source_indexes[x] + len(shortlist_indexes) if x in source_indexes
                            else 0)
                      for x in target]
            prediction = [shortlist_indexes[x] if x in shortlist_indexes
                          else (source_indexes[x] + len(shortlist_indexes) if x in source_indexes
                                else 0)
                          for x in prediction]

            pad_with_zeros(source)
            pad_with_zeros(target)
            pad_with_zeros(prediction)

            source = ' '.join(str(x) for x in source)
            target = ' '.join(str(x) for x in target)
            prediction = ' '.join(str(x) for x in prediction)

            dket_emb_file.write(f"{source}\t{target}\t{prediction}\n")


def convert_fairseq_to_dket(fairseq_path, dket_path):
    with open(fairseq_path) as fairseq_file, open(dket_path, "w") as dket_file:
        while True:
            source = next(fairseq_file, None)
            target = next(fairseq_file, None)
            hypothesis = next(fairseq_file, None)
            detokenized_hypothesis = next(fairseq_file, None)
            score = next(fairseq_file, None)

            if source is None:
                break

            source = source.split("\t")
            target = target.split("\t")
            hypothesis = hypothesis.split("\t")
            detokenized_hypothesis = detokenized_hypothesis.split("\t")
            score = score.split("\t")

            if len(set([line[0].split("-")[-1] for line in [source, target, hypothesis, detokenized_hypothesis, score]])) != 1:
                print("Something went wrong")

            source = source[1:][0].strip()
            target = target[1:][0].strip()
            hypothesis = hypothesis[2:][0].strip()
            detokenized_hypothesis = detokenized_hypothesis[2:][0].strip()
            score = score[1:][0].strip()

            dket_file.write(f"{source}\t{target}\t{hypothesis}\n")


if __name__ == "__main__":
    fairseq_results = r"C:\Users\frank\GitHub\dket\.tests\fairseq\joined_75\results_75_joined.tsv"
    dket_results = r"C:\Users\frank\GitHub\dket\.tests\fairseq\joined_75\results_75_joined_dket.tsv"
    dket_emb_results = r"C:\Users\frank\GitHub\dket\.tests\fairseq\joined_75\results_75_joined_dket_emb.tsv"
    vocabulary_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\vocabulary.idx"
    shortlist_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\shortlist.idx"
    convert_fairseq_to_dket(fairseq_results, dket_results)
    coonvert_dket_to_embeddings(vocabulary_fp, shortlist_fp, dket_results, dket_emb_results)

    # dump_fp = r"C:\Users\frank\GitHub\dket\.tests\2k-open-x-ref--lr04\eval\dump\dump-20001.tsv"
    # report_fp = r"C:\Users\frank\GitHub\dket\.tests\2k-open-x-ref--lr04\2k-open-x-ref--lr04.report.v2.txt"
    report_fp = r"C:\Users\frank\GitHub\dket\.tests\fairseq\joined_75\report.txt"
    analytics.create_report(dket_emb_results, vocabulary_fp, shortlist_fp, report_fp, True)
