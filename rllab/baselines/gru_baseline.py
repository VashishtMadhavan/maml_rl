import numpy as np
from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.regressors.gru_regressor import GRURegressor


class GRUBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            hidden_dim=256,
            num_seq_inputs=1,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GRUBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GRURegressor(
            input_shape=(env_spec.observation_space.flat_dim,),
            hidden_dim=hidden_dim,
            output_dim=1,
            name="vf",
            **regressor_args
        )

    @overrides
    def fit(self, paths, log=True):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)), log=log)

    @overrides
    def predict(self, path):
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
