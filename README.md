# Replication Package for paper "Testing of Deep Reinforcement Learning Agents with Surrogate Models". 

This Readme provides instructions on how to train agents for the three tasks from scratch and how to test them using the codebase.

## 1. Configuration

The commands below assume you are in a Unix system. However, the simulators for the environments used in this project are available for MacOs, Linux and Windows.

Clone this repository:

`git clone https://github.com/matteobiagiola/drl-testing-experiments ~/drl-testing-experiments`

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Create a virtual environment: `conda create -n drl_testing python=3.6`

### 1.1 Mujoco

Before installing the requirements download [Mujoco 2.0](https://roboti.us/download.html) for your system and extract its content in your home directory under `.mujoco`. Then download the [license file](https://roboti.us/file/mjkey.txt) and place it on the previously created `.mujoco` folder in your home directory.

### 1.2 Install requirements

Type `cd ~/drl-testing-experiments && conda activate drl_testing` to activate the environment. Then type `pip install -r requirements.txt` to install the requirements for this project.

### 1.3 Donkey

Download the [Donkey simulator](https://drive.google.com/file/d/1FWnKOYp-xvc9BMv_WfsA1KLewL5Xm9v6/view?usp=sharing) for your system.

On MacOs type `codesign -s --f --deep ~/Desktop/DonkeySimMac/donkey_sim.app` to sign the application (assuming the DonkeySimMac folder is under the Desktop folder).

On Unix system assign the right permission to the application by typing:

`chmod -R 755 ~Desktop/DonkeySimMac/donkey_sim.app` if you are on MacOs

`chmod -R 755 ~Desktop/DonkeySimLinux/donkey_sim.x86_64` if you are on Linux

## 2. Training agents

To activate the environment type `cd ~/drl-testing-experiments && conda activate drl_testing`;

Type 

```
python -m indago.train --algo her -tb logs/tensorboard \
	--seed 2646669604 --env-name park --env-id parking-v0
``` 
to train the Parking agent with the default hyperparameters (`hyperparams/indago/her.yml`);

Type 

```
python -m indago.train --algo tqc -tb logs/tensorboard \
	--seed 2646669604 --env-name humanoid --env-id Humanoid-v0
``` 	
to train the Humanoid agent with the default hyperparameters (`hyperparams/indago/tqc.yml`);

Type 

```
python -m indago.train --algo sac -tb logs/tensorboard \
	--seed 2646669604 --env-name donkey \
	--env-id DonkeyVAE-v0 --log-interval 1000 \
	--exe-path </path/to/donkey-sim-mac/donkey_sim.app> \
	--vae-path logs/generated_track/vae-64.pkl --z-size 64 \
	--simulation-mul 5 --headless
``` 
to train the Donkey agent with the default hyperparameters (`hyperparams/indago/sac.yml`). The `--exe-path` option needs to point to the path to the simulator binary. The `--z-size` option value needs to match the latent space size of the VAE (i.e. 64 in this case). The `--simulation-mul` option tells the Unity engine to make the simulation 5 times faster than real time (values higher than 5 lead to inconsistent training runs and it is not recommended). The self-driving agent trained with 5x does not drive well in real time and viceversa. The `--headless` option makes the simulator headless, i.e. the simulator GUI does not spawn. This has the advantage of consuming less graphics memory but it will not train faster

### 2.1 Tensorboard Logs

To activate the environment type `cd ~/drl-testing-experiments && conda activate drl_testing`;

Type `tensorboard --logdir logs/tensorboard/<env-id>` to see the metrics tracked during training of the DRL agents where: 

`<env-id> = parking-v0 | Humanoid-v0 | DonkeyVAE-v0-scene-generated_track`

### 2.2 Example Train Parking Agent

Let's say we want to train a new Parking agent. The command we need to type is:

```
python -m indago.train --algo her -tb logs/tensorboard \
	--seed 2646669604 --env-name park \
	--env-id parking-v0 \
	--n-timesteps 50000
``` 
This will train a Parking agent for 50000 timesteps (1/3 of the default number of timesteps) and takes about 1.5 hours on a MacBook Pro. Moreover, the command will create a directory `logs/her/parking-v0_1` in your project where environment configurations and models will be saved. Since the `-tb` option is provided, we can look at the training metrics in tensorboard.

To do that head over to another terminal window and type:

```
cd ~/drl-testing-experiments && conda activate drl_testing
tensorboard --logdir logs/tensorboard/parking-v0
```

Open your browser at `localhost:6006`. On the left-hand side of the screen look for `Runs` and select only `HER_1` to see the metrics for the current agent. You might also want to click on settings (gear on the right-hand side of the screen) and check the `Reload data` option. The panel has two sections, namely `rollout` and `time`. The latter shows how fast is the training while the former shows the trend of three metrics over time all averaged over 100 episodes: `ep_len_mean`, i.e. how long is an episode (the lower the better in Parking), `ep_len_rew`, i.e. how much reward the agent is getting in an episode (the higher the better in all cases) and `success_rate`, i.e. the percentage of episodes in which the agent is successful (the higher the better in all cases).

For simplicity the Parking agent trained with the command mentioned above is available [here](https://www.dropbox.com/sh/o5wczg0oxg6h9on/AAAYiYnnjEn3sOELIOJwemZKa?dl=0). Just unzip the logs folder in the root of the project (i.e., `~/drl-testing-experiments`).

## 3. Training the Classifier

Type `cd ~/drl-testing-experiments && conda activate drl_testing` to activate the environment.

### 3.1 Build Held-out Test Set

Type

```
python -m indago.avf.train --algo <algo-name> \
	--env-name <env-name> --env-id <env-id> \
	--exp-id <exp-id> --avf-policy mlp \
	--build-heldout-test \
	--test-split 0.1 \
	--seed 0
```

to build a heldout test set where:

`<algo-name> = her | tqc | sac` for Parking, Humanoid and Donkey respectively;

`<env-name> = park | humanoid | donkey`;

`<env-id> = parking-v0 | Humanoid-v0 | DonkeyVAE-v0`;

`<exp-id>` is written on the directory `logs/<algo-name>/<env-id>_<exp-id>`. For example for Parking agent trained above we have `logs/her/parking-v0_1`, i.e. `<exp-id> = 1`.

The command will create the file `heldout-set-seed-0-0.1-split-5-filter-cls.npz` in the directory `logs/<algo-name>/<env-id>_<exp-id>`

### 3.2 Example Build Test Set for Parking Agent

Build a heldout test set for the Parking agent trained at step 2. 

```
python -m indago.avf.train --algo her \
	--env-name park --env-id parking-v0 \
	--exp-id 1 --avf-policy mlp \
	--build-heldout-test \
	--test-split 0.1 \
	--seed 0
```
The command will create the file `heldout-set-seed-0-0.1-split-5-filter-cls.npz` in the directory `logs/her/parking-v0_1`.

### 3.3 Train a Classifier

Type:

```
python -m indago.avf.train --algo <algo-name> \
	--env-name <env-name> --env-id <env-id> \
	--exp-id <exp-id> --test-split 0.2 --avf-policy mlp \
	--training-progress-filter <filter> --oversample 0.0 \
	--n-epochs 2000 --learning-rate 3e-4 \
	--batch-size 256 --patience 10 \
	--weight-loss --layers <num-hidden-layers> --seed 0 \
	--heldout-test-file <heldout-test-file>.npz
```

to train a classifier where:

`<algo-name> = her | tqc | sac` for Parking, Humanoid and Donkey respectively;

`<env-name> = park | humanoid | donkey`;

`<env-id> = parking-v0 | Humanoid-v0 | DonkeyVAE-v0`;

`<exp-id>` see Section 3.1;

`<filter> = 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80`;

`<layers> = 1 | 2 | 3 | 4`;

`<heldout-test-file>` created in Section 3.1.


### 3.4 Example Train a Classifier for Parking Agent

Requires Section 3.2 to be completed. The command:

```
python -m indago.avf.train --algo her \
		--env-name park --env-id parking-v0 \
		--exp-id 1 --test-split 0.2 --avf-policy mlp \
		--training-progress-filter 20 --oversample 0.0 \
		--n-epochs 2000 --learning-rate 3e-4 \
		--batch-size 256 --patience 10 \
		--weight-loss --layers 4 --seed 2216495700 \
		--heldout-test-file heldout-set-seed-0-0.1-split-5-filter.npz
```

trains a multi-layer perceptron with 4 hidden layers and filtering the initial 20% of the configurations in `logs/her/parking-v0_1`. If you followed the instructions above for training the Parking agent and the classifier you should obtain a precision of 0.29 and a recall of 0.11 in the test set.

In general, the number of hidden layers and the filtering level are not known a priori and they depend on the case study. In order to get the best classifier possible it is recommended to run different training runs (command above) with different seeds and choose the classifier with the best precision.

## 4. Testing the Agent

Given a trained agent you can test it using random without a trained classifier. If the agent under test is the Donkey agent then at the commands below you need to append the Donkey simulator related parameters, i.e.:

```
--exe-path </path/to/donkey-sim-mac/donkey_sim.app> \
--vae-path logs/generated_track/vae-64.pkl \
--z-size 64 \
--simulation-mul 5 \
--headless
```

Type `cd ~/drl-testing-experiments && conda activate drl_testing` to activate the environment.

### 4.1.1 Random

```
python -m indago.experiments --algo <algo-name> \
		--exp-id <exp-id> --env-name <env-name> --env-id <env-id> \
		--avf-test-policy random --failure-prob-dist \
		--num-episodes <num-episodes> \
		--num-runs-each-env-config <num-runs> \
		--num-runs-experiments <num-runs-experiments>
```
where:

`<algo-name> = her | tqc | sac` for Parking, Humanoid and Donkey respectively;

`<env-name> = park | humanoid | donkey`;

`<env-id> = parking-v0 | Humanoid-v0 | DonkeyVAE-v0`;

`<exp-id>` see Section 3.1;

`<num-episodes>` it is the number of environment configurations to generate;

`<num-runs>` should be 1 if the environment is deterministic (e.g. Parking) while it should be > 1 if the environment is stochastic (e.g. Humanoid and Donkey). It is the number of times the trained agent is executed on each generated environment configuration;

`<num-runs-experiments>` how many times to run the same experiment with different random seeds.

### 4.1.2 Sampling

It requires a trained classifier. For example, if the classifier trained with `<filter> = 20` and `<layers> = 4` is available, then the sampling approach can be applied.

```
python -m indago.experiments --algo <algo-name> \
		--exp-id <exp-id> --env-name <env-name> --env-id <env-id> \
		--avf-test-policy nn --failure-prob-dist \
		--num-episodes <num-episodes> \
		--num-runs-each-env-config <num-runs> \
		--training-progress-filter <filter> \
		--layers <layers> \
		--budget <budget> \
		--num-runs-experiments <num-runs-experiments>
```
where:

`<filter> = 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80`;

`<layers> = 1 | 2 | 3 | 4`;

`<budget>` budget in seconds for the generation of each environment configuration (default = -1);

`<num-runs-experiments>` how many times to run the same experiment with different random seeds.

### 4.1.3 Hill Climbing Methods

It requires a trained classifier. For example, if the classifier trained with `<filter> = 20` and `<layers> = 4` is available, then the hill climbing approaches can be applied.

```
python -m indago.experiments --algo <algo-name> \
		--exp-id <exp-id> --env-name <env-name> --env-id <env-id> \
		--avf-test-policy <hc-policy> --failure-prob-dist \
		--num-episodes <num-episodes> \
		--num-runs-each-env-config <num-runs> \
		--training-progress-filter <filter> \
		--layers <layers> \
		--hc-counter <hc-counter> \
		--neighborhood-size <neighborhood-size> \
		--budget <budget> \
		--num-runs-experiments <num-runs-experiments>
```
where:

`<filter> = 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80`;

`<layers> = 1 | 2 | 3 | 4`;

`<hc-policy> = hc | hc_failure | hc_saliency_rnd | hc_saliency_failure`;

`<hc-counter>` maximum number of neighborhoods to generate. Default 100;

`<neighborhood-size>` number of mutations of the current environment configuration. Default 50.

`<budget>` budget in seconds for the generation of each environment configuration (default = -1);

`<num-runs-experiments>` how many times to run the same experiment with different random seeds.

### 4.1.4 Genetic Algorithms Methods

It requires a trained classifier. For example, if the classifier trained with `<filter> = 20` and `<layers> = 4` is available, then the hill climbing approaches can be applied.

```
python -m indago.experiments --algo <algo-name> \
		--exp-id <exp-id> --env-name <env-name> --env-id <env-id> \
		--avf-test-policy <ga-policy> --failure-prob-dist \
		--num-episodes <num-episodes> \
		--num-runs-each-env-config <num-runs> \
		--training-progress-filter <filter> \
		--layers <layers> \
		--population-size <population-size> \
		--crossover-rate <crossover-rate> \
		--budget <budget> \
		--num-runs-experiments <num-runs-experiments>
```
where:

`<filter> = 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80`;

`<layers> = 1 | 2 | 3 | 4`;

`<ga-policy> = ga_rnd | ga_failure | ga_saliency_rnd | ga_saliency_failure`;

`<population-size>` maximum number of individuals in the population. Default 50;

`<crossover-rate>` probability of crossover. Default 0.75;

`<budget>` budget in seconds for the generation of each environment configuration (default = -1);

`<num-runs-experiments>` how many times to run the same experiment with different random seeds.

### 4.2 Example Testing the Parking Agent

Requires Section 3.4 to be completed. All the commands below will run 10 experiments each with different random seeds.

#### 4.2.1 Random

```
python -m indago.experiments --algo her \
		--exp-id 1 --env-name park --env-id parking-v0 \
		--avf-test-policy random --failure-prob-dist \
		--num-episodes 50 \
		--num-runs-each-env-config 1 \
		--num-runs-experiments 10
```

#### 4.2.2 Sampling

```
python -m indago.experiments --algo her \
        --exp-id 1 --env-name park --env-id parking-v0 \
        --avf-test-policy nn --failure-prob-dist \
        --num-episodes 50 \
        --num-runs-each-env-config 1 \
        --training-progress-filter 20 \
        --layers 4 \
        --budget 3 \
        --num-runs-experiments 10
```

#### 4.2.3 Hill Climbing Saliency Failure

```
python -m indago.experiments --algo her \
        --exp-id 1 --env-name park --env-id parking-v0 \
        --avf-test-policy hc_saliency_failure --failure-prob-dist \
        --num-episodes 50 \
        --num-runs-each-env-config 1 \
        --training-progress-filter 20 \
        --layers 4 \
        --budget 3 \
        --num-runs-experiments 10
```

#### 4.2.4 Genetic Algorithm Saliency Failure

```
python -m indago.experiments --algo her \
        --exp-id 1 --env-name park --env-id parking-v0 \
        --avf-test-policy ga_saliency_failure --failure-prob-dist \
        --num-episodes 50 \
        --num-runs-each-env-config 1 \
        --training-progress-filter 20 \
        --layers 4 \
        --budget 3 \
        --num-runs-experiments 10
```

## 5. Analyze the Results

### 5.1 Analyze Failure Search Results

Requires Section 4. to be completed.

Type `cd ~/drl-testing-experiments && conda activate drl_testing` to activate the environment.

Type:

```
python -m indago.env_logs_analysis_trials \
			--folder <folder-name> \
			--env-name <env-name> 
			--names <[list-of-names]>
```
where:

`<folder-name>` folder where logs are;

`<env-name> = park | humanoid | donkey`;

`<[list-of-names]>` where each name `random | nn | hc | hc_failure | hc_saliency_rnd | hc_saliency_failure | ga_rnd | ga_failure | ga_saliency_rnd | ga_saliency_failure` and the logs of each technique needs to be present in the folder;

The command perfoms a statistical analysis of the failures found by each technique by computing and printing p-value and effect size for each pair of techniques.

#### 5.1.1 Example Analyze Failure Search Results for the Parking Agent

Requires Section 4.2 to be completed.

```
python -m indago.env_logs_analysis_trials \
			--folder logs/her/parking-v0_1 \
			--env-name park
			--names random nn hc_saliency_failure ga_saliency_failure
```

### 5.2 Analyze Diversity Results

Requires Section 4. to be completed.

Type `cd ~/drl-testing-experiments && conda activate drl_testing` to activate the environment.

Type:

```
python -m indago.experiments --algo <algo-name> \
		--exp-id <exp-id> --env-name <env-name> --env-id <env-id> \
		--avf-test-policy replay_test_failure --failure-prob-dist \
		--exp-file <exp-file> \
```
where:

`<exp-file>` is a log file starting with `testing-*-trial.txt` in the logs folder. The command will parse the file to look for environment configurations that caused a failure at testing time and replay those configurations (`replay_test_failure`).

Afterwards, diversity can be computed by typing:

```
python -m indago.diversity \
		--folder <folder-name> \
		--env-name <env-name> \
		--pattern 
		--names <[list-of-names]>  \
		--type <type> \
		--sil-threshold 20 \
		--visualize
```

where:

`<folder-name>` folder where logs are;

`<env-name> = park | humanoid | donkey`;

`<[list-of-names]>` where each name `random | nn | hc | hc_failure | hc_saliency_rnd | hc_saliency_failure | ga_rnd | ga_failure | ga_saliency_rnd | ga_saliency_failure` and the logs of each technique needs to be present in the folder;

`<type> = input | output`;

The command computes coverage and entropy for the techniques provided. Morover, it plots the projection in 2D of the clusters.

#### 5.2.1 Example Analyze Analyze Diversity Results for the Parking Agent

Requires Section 5.1.1 to be completed.

Type:

```
python -m indago.experiments \
		--algo her \
		--exp-id 1 \
		--env-name park \
		--env-id parking-v0 \
		--avf-test-policy replay_test_failure \
		--failure-prob-dist \
		--num-runs-each-env-config 1 \
		--exp-file testing-random-failure-prob-dist-50-1-0-trial.txt
```

for replaying failures generated by `random`.

Type:

```
python -m indago.experiments \
		--algo her \
		--exp-id 1 \
		--env-name park \
		--env-id parking-v0 \
		--avf-test-policy replay_test_failure \
		--failure-prob-dist \
		--num-runs-each-env-config 1 \
		--exp-file testing-mlp-nn-original-20-0.0-4-100000-failure-prob-dist-50-1-budget-3-0-trial.txt
```

for replaying failures generated by `nn` (i.e., sampling).

Type:

```
python -m indago.experiments \
		--algo her \
		--exp-id 1 \
		--env-name park \
		--env-id parking-v0 \
		--avf-test-policy replay_test_failure \
		--failure-prob-dist \
		--num-runs-each-env-config 1 \
		--exp-file testing-mlp-hc_saliency_failure-30-0.0-4-50-100-0.005-failure-prob-dist-50-1-budget-3-0-trial.txt
```

for replaying failures generated by `hc_saliency_failure`.

Type:

```
python -m indago.experiments \
		--algo her \
		--exp-id 1 \
		--env-name park \
		--env-id parking-v0 \
		--avf-test-policy replay_test_failure \
		--failure-prob-dist \
		--num-runs-each-env-config 1 \
		--exp-file testing-mlp-ga_saliency_failure-30-0.0-4-50-0.75-failure-prob-dist-50-1-budget-3-0-trial.txt
```

for replaying failures generated by `ga_saliency_failure`.

Then type:

`
python -m indago.diversity \
			--folder logs/her/parking-v0_1 \
			--env-name park \
			--pattern \
			--names random nn hc_saliency_failure ga_saliency_failure \
			--type input \
			--sil-threshold 20 \
			--visualize
`

to compute input coverage and entropy and type:

`
python -m indago.diversity \
			--folder logs/her/parking-v0_1 \
			--env-name park \
			--pattern \
			--names random nn hc_saliency_failure ga_saliency_failure \
			--type output \
			--sil-threshold 20 \
			--visualize
`

to compute output coverage and entropy.

