from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_oracle import AntEnvOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import os
import pickle
import tensorflow as tf
import argparse

stub(globals())


parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, help='file to load for testing')
parser.add_argument('--save_name', type=str, help='save name for MAML experiment')
parser.add_argument('--video', action='store_true', help='to save video')
args = parser.parse_args()

if not make_video:
    test_num_goals = 40
    np.random.seed(1)
    goals = np.random.uniform(0.0, 3.0, size=(test_num_goals, ))
else:
    np.random.seed(1)
    test_num_goals = 2
    goals = [0.0, 3.0]
    file_ext = 'mp4'  # can be mp4 or gif
print(goals)


exp_names = [args.save_name + '_maml']
step_sizes = [0.1, 0.2, 1.0, 0.0]
initial_params_files = [str(args.test_file)]
all_avg_returns = []

for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
    avg_returns = []
    for goal in goals:
        env = TfEnv(normalize(AntEnvRand()))
        n_itr = 4
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = VPG(
            env=env,
            policy=None,
            load_policy=initial_params_file,
            baseline=baseline,
            batch_size=4000,  # 2x
            max_path_length=200,
            n_itr=n_itr,
            reset_arg=goal,
            optimizer_args={'init_learning_rate': step_sizes[step_i], 'tf_optimizer_args': {'learning_rate': 0.5*step_sizes[step_i]}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
        )

        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=1,
            exp_prefix=args.save_name,
            exp_name='test',
            #plot=True,
        )

        # get return from the experiment
        with open('data/local/{}/test/progress.csv'.format(args.save_name), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            row = None
            returns = []
            for row in reader:
                i+=1
                if i ==1:
                    ret_idx = row.index('AverageReturn')
                else:
                    returns.append(float(row[ret_idx]))
            avg_returns.append(returns)

        if make_video:
            data_loc = 'data/local/{}/test/'.format(args.save_name)
            save_loc = 'data/local/{}/test/'.format(args.save_name)
            param_file = initial_params_file
            save_prefix = save_loc + names[step_i] + '_goal_' + str(goal)
            video_filename = save_prefix + 'prestep.' + file_ext
            os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=300 --video_filename='+video_filename)
            for itr_i in range(3):
                param_file = data_loc + 'itr_' + str(itr_i)  + '.pkl'
                video_filename = save_prefix + 'step_'+str(itr_i)+'.'+file_ext
                os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=300 --video_filename='+video_filename)


    all_avg_returns.append(avg_returns)
    task_avg_returns = []
    for itr in range(len(all_avg_returns[step_i][0])):
        task_avg_returns.append([ret[itr] for ret in all_avg_returns[step_i]])

    if not make_video:
        results = {'task_avg_returns': task_avg_returns}
        with open(exp_names[step_i] + '.pkl', 'wb') as f:
            pickle.dump(results, f)

for i in range(len(initial_params_files)):
    returns = []
    std_returns = []
    returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
    std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))
    print(initial_params_files[i])
    print(returns)
    print(std_returns)



