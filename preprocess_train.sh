TEXT=bpe.32k
fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt18_de_en --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

sed -r -i s/(([^ ]* ){2}).*/\1/' bpe.32k/codes

mkdir -p checkpoints/fconv_iwslt_de_en
fairseq-train \
    data-bin/wmt18_de_en \
    --arch fconv_iwslt_de_en \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/fconv_iwslt_de_en