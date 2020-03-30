#!/usr/bin/env bash
#
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=5000

URL="https://raw.githubusercontent.com/frankilepro/dket/master/datasets/def-form.tgz"
GZ=def-form.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=def
tgt=form
lang=def-form
prep=dket.tokenized.def-form
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

# echo "pre-processing train data..."
# for l in $src $tgt; do
#     f=train.$l
#     tok=train.tok.$l

#     cat $orig/$lang/$f | perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
#     echo ""
# done
# perl $CLEAN -ratio 1.5 $tmp/train.tok $src $tgt $tmp/train.clean 1 175
# for l in $src $tgt; do
#     perl $LC < $tmp/train.clean.$l > $tmp/train.$l
# done

echo "pre-processing train/valid/test data..."
for l in $src $tgt; do
    for o in "train" "valid" "test"; do
    f=$o.$l
    tok=$o.tok.$l
    
    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $tmp/$tok
    cp $orig/$lang/$f $tmp/$f
    echo ""
    done
done


# echo "creating train, valid, test..."
# for l in $src $tgt; do
#     # awk '{if (NR%23 == 0)  print $0; }' $tmp/valid.$l > $tmp/valid.$l
#     # awk '{if (NR%23 != 0)  print $0; }' $tmp/train.$l > $tmp/train.$l
#     # awk '{if (NR%23 != 0)  print $0; }' $tmp/test.$l > $tmp/test.$l
#     for o in "train" "valid" "test"; do
#         f=$o.$l
#         cp $orig/$lang/$f $tmp/$f
#     done
# done

TRAIN=$tmp/train
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
