#!/bin/bash
# The large swap dataset: fetch RAVEL (github.com/explanare/ravel, data.tgz) and the LRE
# relations (github.com/evandez/relations, data/), build the dataset with the hand-written
# templates, train the band maps on it from the plain J-lens and from the identity, then score
# swap success on the held-out relations, probe-swap and recall@10 next to the baselines.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=c2204,c2205,c2206,c2207,d1002,d1007,d1009,d1010,d1011,d1012,d1013,d1015,d1017,d1019,d1020,d1022,d1025,d1027
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J ilens-swapdata2
#SBATCH -o logs/intervention_swap_dataset_v2_%j.log
set -euo pipefail
source /projects/RohanKathuria/jpp_lens/slurm/common.sh
I=$OUT/intervention
D=$I/swap_data_v2
X=data_ext
CSV=$OUT/model_correctness_stripped.csv
STEPS=${STEPS:-6000}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== fetch $(date +%T)"
mkdir -p $X/ravel
[ -d $X/relations ] || git clone --depth 1 https://github.com/evandez/relations $X/relations
[ -f $X/ravel/data.tgz ] || curl -sSL -o $X/ravel/data.tgz https://github.com/explanare/ravel/raw/main/data.tgz
[ -n "$(find $X/ravel -name '*.json' | head -1)" ] || tar -xzf $X/ravel/data.tgz -C $X/ravel
find $X/ravel -name '*.json' | sort | head -30
ls $X/relations/data/*/ | head -80
python3 -c "import json,glob; f=sorted(glob.glob('$X/relations/data/factual/*.json'))[0]; d=json.load(open(f)); print(f, list(d)); print({k: d[k] for k in d if k != 'samples'}); print(d['samples'][:2])"

echo "== build dataset $(date +%T)"
[ -f $D/train.json ] || $PY experiments/intervention_lens/build_swap_dataset_v2.py --ravel-dir $X/ravel \
    --lre-dir $X/relations/data --max-heldout-items 600 --out-dir $D 2>&1 | grep --line-buffered -v httpx

echo "== fits $(date +%T)"
for INIT in jlens identity; do
    INIT_PATH=$([ $INIT = jlens ] && echo $I/plain_jlens64.pt || echo identity)
    [ -f $I/swapdata2_$INIT.pt ] || $PY scripts/jpp_cli.py fit-swap --hf-model-name $MODEL --layers $LAYERS \
        --mode supervised --swap-data $D/train.json --init $INIT_PATH --steps $STEPS --eval-every 250 \
        --patience 1000 --heldout-examples 512 --monitor-data $D/heldout.json --checkpoint-name swapdata2_$INIT --out $I/swapdata2_$INIT.pt \
        2>&1 | grep --line-buffered -v httpx
done

LENSES="$I/swapdata2_jlens.pt $I/swapdata2_identity.pt $I/swapdata_jlens.pt $I/swapdata_identity.pt \
    $I/plain_jlens64.pt $OUT/jpp_lens_v2.pt $OUT/pooled_lens.pt logit"
echo "== evaluate $(date +%T)"
$PY scripts/jpp_cli.py evaluate --hf-model-name $MODEL --layers $LAYERS --correctness-csv $CSV \
    --lens $LENSES --out $I/eval_held_out_swapdata2.csv 2>&1 | grep --line-buffered -v -e httpx -e "scored item"
echo "== probe-swap $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS --lens $LENSES \
    --out $I/swapdata2_probe_swap.parquet 2>&1 | grep --line-buffered -v -e httpx -e "ran trial"
echo "== swap success on held-out relations $(date +%T)"
$PY scripts/jpp_cli.py probe-swap --hf-model-name $MODEL --layers $LAYERS --lens $LENSES \
    --data $D/heldout.json --out $I/swapdata2_heldout_swap.parquet 2>&1 | grep --line-buffered -v -e httpx -e "ran trial"
echo "SWAP_DATASET_V2_DONE $(date +%T)"
