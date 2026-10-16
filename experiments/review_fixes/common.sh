# Shared settings for the review-fixes jobs. They run from a separate copy of this branch,
# /projects/RohanKathuria/jpp_lens_review, so jobs started from the main copy keep their code.
# artifacts/, results/ and data_ext/ there are links to the main copy's; new outputs go to
# artifacts/gemma4_e2b/review_fixes/. The venv is the main copy's; PYTHONPATH puts this
# branch's src/ ahead of the venv's editable install of the main copy.
cd /projects/RohanKathuria/jpp_lens_review
source /projects/RohanKathuria/param-clustering/env.sh
export PYTHONPATH=$PWD/src
export PY="/projects/RohanKathuria/jpp_lens/.venv/bin/python -u"
export MODEL=google/gemma-4-E2B
export LAYERS=4,8,12,16,20,24,28
export OUT=artifacts/gemma4_e2b
export I=$OUT/intervention
export R=$OUT/review_fixes
export CSV=$OUT/model_correctness_stripped.csv
mkdir -p $R logs
# Drop only the HTTP request lines; progress lines stay.
quiet() { grep --line-buffered -v -e httpx -e "dropping swap item"; }
