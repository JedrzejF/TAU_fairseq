from fairseq.models.transformer import TransformerModel
import torch
from tqdm import tqdm

de2en = TransformerModel.from_pretrained(
  'checkpoints/fconv_wmt_en_de',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/wmt17_en_de',
  bpe='subword_nmt',
  bpe_codes='bpe.32k/codes'
)
for folder_name in ['dev-0', 'dev-1', 'test-A']:
    num_lines = sum(1 for line in open(f'{folder_name}/in.tsv'))
    with open(f'{folder_name}/in.tsv', 'r') as in_file, open(f'{folder_name}/out.tsv', 'w+') as out_file:
        for line in tqdm(in_file, total=num_lines):
            translation = de2en.translate(line)
            out_file.write(f'{translation}\n')