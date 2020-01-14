
WMT2017 German-English machine translation challenge for news
=============================================================

Translate news articles from German into English.

This is WMT2017 news challenge reformatted as a Gonito.net challenge,
all the data were taken from <http://www.statmt.org/wmt17/translation-task.html>.

BLEU is used as the evaluation metric.

Directory structure
-------------------

* `README.md` — this file
* `config.txt` — configuration file
* `train/` — directory with training data
* `train/commoncrawl.tsv.xz` — Common Crawl parallel corpus
* `train/europarl-v7.tsv.xz` — Europarl parallel corpus
* `train/news-commentary-v12.tsv.xz` — News Commentary parallel corpus
* `train/rapid2016.tsv.xz` — Rapid corpus of EU press releases
* `dev-0/` — directory with dev (test) data (WMT2015 test set)
* `dev-0/in.tsv` — German input text for the dev set
* `dev-0/expected.tsv` — English reference translation for the dev set
* `dev-1/` — directory with dev (test) data (WMT2016 test set)
* `dev-1/in.tsv` — German input text for the dev set
* `dev-1/expected.tsv` — English reference translation for the dev set
* `test-A` — directory with test data
* `test-A/in.tsv` — German input data for the test set (WMT2017 test set)
* `test-A/expected.tsv` — English reference translation for the test set

Training sets
-------------

All training sets were compressed with xz, use `xzcat` to decompress:

    $ xzcat train/*.tsv.xz | ...

The pairs where German or English side is empty were removed from the
training sets.

### Stats

    $ xzcat train/*.tsv.xz | wc
    5906167 247901483 1680105853

    $ xzcat train/*.tsv.xz | cut -f 1 | wc -m
    876313942

    $ xzcat train/*.tsv.xz | cut -f 2 | wc -m
    787047585

Test sets
---------

Reference English translations in the dev and test sets were tokenised
using Moses MT tokeniser (without HTML escaping).

Monolingual data
----------------

Monolingual data was not included here.
