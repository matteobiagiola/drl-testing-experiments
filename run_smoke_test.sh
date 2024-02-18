#!/bin/bash

source ~/.bashrc

# Set the path to your conda installation
conda_path=$(which conda | sed 's/\/bin\/conda//')

# Source the conda.sh script to make conda commands available
source "$conda_path/etc/profile.d/conda.sh"

conda activate indago

env_name=

while [ $# -gt 0 ] ; do
  case $1 in
    -e | --env-name) env_name="$2" ;;
  esac
  shift
done

if [[ "$env_name" == "humanoid" ]]; then
    python -m indago.train --algo tqc -tb logs/tensorboard --seed 2646669604 \
        --env-name humanoid --env-id Humanoid-v0 --no-log --n-timesteps 1500
elif [[ "$env_name" == "parking" ]]; then
    python -m indago.train --algo her -tb logs/tensorboard --seed 2646669604 \
        --env-name park --env-id parking-v0 --no-log --n-timesteps 150
elif [[ "$env_name" == "donkey" ]]; then
    xvfb-run --auto-servernum --server-args='-screen 0 1920x1080x24' \
        python -m indago.train --algo sac -tb logs/tensorboard --seed 2646669604 \
            --env-name donkey --env-id DonkeyVAE-v0 --no-log --n-timesteps 150 \
            --exe-path /root/DonkeySimLinuxIndago/donkey_sim.x86_64 \
            --vae-path logs/generated_track/vae-64.pkl \
            --z-size 64 --simulation-mul 5 --headless
else
  echo Env name "$env_name" not supported
  exit 1
fi

