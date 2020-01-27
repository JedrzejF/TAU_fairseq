src=de
tgt=en
lang=de-en
prep=wmt18_de_en
tmp=$prep/tmp
orig=orig
dev=dev/newstest2012
codes=32000
bpe=bpe.32k

mkdir -p $tmp $prep $bpe

mkdir train
mv $tmp/{train,valid,test}.{$src,$tgt} train

#BPE
git clone https://github.com/glample/fastBPE.git
pushd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
popd
fastBPE/fast learnbpe $codes output/train.$src output/train.$tgt > $bpe/codes
for split in {train,valid,test}; do for lang in {de,en}; do fastBPE/fast applybpe $bpe/$split.$lang output/$split.$lang $bpe/codes; done; done
