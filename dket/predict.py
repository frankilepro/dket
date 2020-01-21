from dket import analytics
from dket import runtime
from dket import create_rio
import os
import tempfile

# dump_fp = r"C:\Users\frank\GitHub\dket\.tests\2k-open-x-ref--lr04\eval\dump\dump-20001.tsv"
# vocabulary_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\vocabulary.idx"
# shortlist_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\shortlist.idx"
# report_fp = r"C:\Users\frank\GitHub\dket\.tests\2k-open-x-ref--lr04\2k-open-x-ref--lr04.report.txt"
# analytics.create_report(dump_fp, vocabulary_fp, shortlist_fp, report_fp, True)


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

    vocabulary_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\vocabulary.idx"
    shortlist_fp = r"C:\Users\frank\GitHub\dket\datasets\2k-open-x-ref\shortlist.idx"
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
