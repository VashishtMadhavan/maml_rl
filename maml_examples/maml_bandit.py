from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_examples.random_bandit import RandomBanditEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_categorical_mlp_policy import MAMLCategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=50)
parser.add_argument('--n', type=int, default=500)
parser.add_argument('--iters', type=int, default=400)
parser.add_argument('--expt_name', type=str, default='bandit_debug')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

stub(globals())

env = TfEnv(normalize(RandomBanditEnv(k=args.k, n=args.n)))

policy = MAMLCategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    grad_step_size=0.1,
    hidden_nonlinearity=tf.nn.relu,
    hidden_sizes=(100, 100),
)
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = MAMLTRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=20, # number of trajs for grad update
    max_path_length=int(args.n),
    meta_batch_size=40,
    num_grad_updates=1,
    n_itr=args.iters,
    use_maml=True,
    step_size=0.01
)

run_experiment_lite(
    algo.train(),
    exp_prefix=args.expt_name,
    exp_name='run_{}'.format(args.seed),
    n_parallel=1,
    snapshot_mode="gap",
    snapshot_gap=20,
    python_command='python3',
    seed=args.seed,
    mode="local",
    # plot=True,
)
