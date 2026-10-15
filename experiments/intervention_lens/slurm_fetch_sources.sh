#!/bin/bash
# Fetch the extra swap-data sources into data_ext/ and convert them to relation files
# (data_ext/relations_extra/*.json): CounterFact (rome.baulab.info), PARAREL
# (github.com/yanaiela/pararel), MUSE en-fr/de/es dictionaries (dl.fbaipublicfiles.com) and
# WordNet (nltk data). CPU job.
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J swapdata-fetch
#SBATCH -o logs/swapdata_fetch_%j.log
set -euo pipefail
cd /projects/RohanKathuria/jpp_lens
X=data_ext
mkdir -p $X/counterfact $X/muse $X/nltk_data
[ -f $X/counterfact/counterfact.json ] || curl -sSL -o $X/counterfact/counterfact.json \
    https://rome.baulab.info/data/dsets/counterfact.json
[ -d $X/pararel ] || git clone --depth 1 -q https://github.com/yanaiela/pararel $X/pararel
for L in fr de es; do
    [ -f $X/muse/en-$L.txt ] || curl -sSL -o $X/muse/en-$L.txt https://dl.fbaipublicfiles.com/arrival/dictionaries/en-$L.txt
done
# WordNet straight from the NLTK data repository (nltk's own downloader refuses to fetch
# through the cluster's proxy).
export NLTK_DATA=$PWD/$X/nltk_data
mkdir -p $NLTK_DATA/corpora
[ -d $NLTK_DATA/corpora/wordnet ] || { curl -sSL -o $NLTK_DATA/corpora/wordnet.zip \
    https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip \
    && (cd $NLTK_DATA/corpora && unzip -q -o wordnet.zip); }
du -sh $X/counterfact $X/pararel $X/muse $X/nltk_data
CONVERT="$HOME/.local/bin/uv run --no-project --python 3.12 --with nltk python experiments/intervention_lens/convert_extra_sources.py --out-dir $X/relations_extra"
$CONVERT --counterfact $X/counterfact/counterfact.json || echo "counterfact conversion failed"
$CONVERT --pararel $X/pararel || echo "pararel conversion failed"
$CONVERT --muse-dir $X/muse || echo "muse conversion failed"
$CONVERT --wordnet || echo "wordnet conversion failed"
ls -la $X/relations_extra
echo "FETCH_DONE $(date +%T)"
