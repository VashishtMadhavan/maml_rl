from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_categorical_mlp_policy import MAMLCategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--expt_name', type=str, default='bandit_debug')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

stub(globals())
env_name = "Bandit_k{}_n{}-v0".format(args.k, args.n)

env = TfEnv(normalize(GymEnv(env_name, record_video=False, record_log=False)))

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
    n_itr=800,
    use_maml=True,
    step_size=0.01
)

run_experiment_lite(
    algo.train(),
    exp_prefix=args.expt_name,
    exp_name='run_{}'.format(args.seed),
    n_parallel=8,
    snapshot_mode="gap",
    snapshot_gap=100,
    python_command='python3',
    seed=args.seed,
    mode="local",
    # plot=True,
)
