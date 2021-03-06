from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_var', type=str, default='pos', help='either [pos, direc, vel]')
parser.add_argument('--use_maml', type=int, default=0)
parser.add_argument('--expt_name', type=str, default='debug')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

stub(globals())

if args.task_var == 'direc':
    env = TfEnv(normalize(AntEnvRandDirec()))
elif args.task_var == 'vel':
    env = TfEnv(normalize(AntEnvRand()))
elif args.task_var == 'pos':
    env = TfEnv(normalize(AntEnvRandGoal()))

policy = MAMLGaussianMLPPolicy(
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
    max_path_length=200,
    meta_batch_size=40,
    num_grad_updates=1,
    n_itr=800,
    use_maml=bool(args.use_maml),
    step_size=0.01,
    plot=False,
)

run_experiment_lite(
    algo.train(),
    exp_prefix=args.expt_name,
    exp_name='run_{}'.format(args.seed),
    n_parallel=8,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="gap",
    snapshot_gap=25,
    python_command='python3',
    seed=args.seed,
    mode="local",
    # plot=True,
    # terminate_machine=False,
)
