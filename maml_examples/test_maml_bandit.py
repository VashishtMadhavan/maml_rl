from maml_examples.random_bandit import RandomBanditEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='parameter file')
parser.add_argument('--num_goals', type=int, default=40)
parser.add_argument('--k', type=int, default=50)
parser.add_argument('--n', type=int, default=500)
parser.add_argument('--step_size', type=float, default=0.1)
args = parser.parse_args()
n_itr = 5

stub(globals())

goals = np.random.uniform(0.0, 1.0, size=(args.num_goals, test_k))
avg_returns = []
for goal in goals:
    goal = list(goal)

    env = TfEnv(normalize(RandomBanditEnv(k=args.k, n=args.n, goal=goal)))
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    if not args.path:
        raise Exception("Please enter a valid path for the parameter file")

    algo = VPG(
        env=env,
        policy=None,
        load_policy=args.path,
        baseline=baseline,
        batch_size=args.n*args.num_goals, 
        max_path_length=args.n,
        n_itr=n_itr,
        optimizer_args={'init_learning_rate': args.step_size, 'tf_optimizer_args': {'learning_rate': 0.5*args.step_size}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        snapshot_mode="last",
        seed=4,
        exp_prefix='trpobandit_test',
        exp_name='test',
    )
    # get return from the experiment
    with open('../data/local/trpobandit-test/test/progress.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        row = None
        returns = []
        for row in reader:
            i+=1
            if i ==1:
                col_idx = row.index("AverageReturn")
            else:
                returns.append(float(row[col_idx]))
        avg_returns.append(returns)

avg_returns = np.array(avg_returns)
for itr in range(n_itr):
    print("Step {} Stats".format(itr))
    print("**********************")
    print("Mean Return: {}".format(np.mean(avg_returns[:, itr])))
    print("Std Return: {}".format(np.std(avg_returns[:, itr])))

