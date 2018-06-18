from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gru_baseline import 
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer 
from rllab.misc.instrument import stub, run_experiment_lite
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help='env to test on')
parser.add_argument("--expt_name", type=str, help='experiment name')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gamma", type=float, default=0.99, help='discount factor')
parser.add_argument("--lamb", type=float, default=0.99, help='for GAE')
parser.add_argument("--n_itr", type=int, default=1000, help='num training iters')
parser.add_argument("--hidden", type=int, default=256, help='num hidden units')
parser.add_argument("--batch_size", type=int, default=250000, help='batch size')
parser.add_argument("--gru_base", action='store_true')
args = parser.parse_args()

stub(globals())

# TODO: get a meta-learning env
env = TfEnv(normalize(CartpoleEnv()))

policy = CategoricalGRUPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_dim=args.hidden,
    hidden_nonlinearity=tf.nn.relu,
    state_include_action=True,
    gru_layer_cls=L.TfGRULayer,
)

# TODO: get a GRU baseline 
if args.gru_base:
    baseline = LinearFeatureBaseline(env_spec=env.spec)
else:
    baseline = GRUBaseline(env_spec=env.spec, hidden_dim=args.hidden)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=args.batch_size,
    max_path_length=100,
    n_itr=args.n_itr,
    gae_lambda=args.lamb,
    discount=args.gamma,
    step_size=0.01,
    optimizer=FirstOrderOptimizer(batch_size=None, max_epochs=1)
)
run_experiment_lite(
    algo.train(),
    exp_prefix=args.expt_name,
    exp_name='run_{}'.format(args.seed),
    snapshot_mode='gap',
    snapshot_gap=100,
    n_parallel=4,
    seed=args.seed,
)
