cuda="$1"
config="$2"

if [[ $# -ne 2 ]] ; then
  echo 'Usage: sweep.sh <cuda> <config>'
  exit 1
fi


python run_experiment.py -cn="$config" link=false cuda="$cuda" learning_rate=5e-6,1e-5,5e-5 label_smoothing_factor=0.0,0.05,0.1,0.2 weight_decay=0.0,0.05,0.1 -m
