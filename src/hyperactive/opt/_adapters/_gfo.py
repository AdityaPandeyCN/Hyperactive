"""Adapter for gfo package."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from skbase.utils.stdout_mute import StdoutMute

from hyperactive.base import BaseOptimizer

__all__ = ["_BaseGFOadapter"]


class _BaseGFOadapter(BaseOptimizer):
    """Adapter base class for gradient-free-optimizers.

    * default tag setting
    * default _run method
    * default get_search_config
    * default get_test_params
    * Handles defaults for "initialize" parameter
    * extension interface: _get_gfo_class, docstring, tags
    """

    _tags = {
        "authors": "SimonBlanke",
        "python_dependencies": ["gradient-free-optimizers>=1.5.0"],
    }

    def __init__(self):
        super().__init__()

        self._union_grid_configs = None

        if self.initialize is None:
            self._initialize = {"grid": 4, "random": 2, "vertices": 4}
        else:
            self._initialize = self.initialize

    def _get_gfo_class(self):
        """Get the GFO class to use.

        Returns
        -------
        class
            The GFO class to use. One of the concrete GFO classes
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def get_search_config(self):
        """Get the search configuration.

        Returns
        -------
        dict with str keys
            The search configuration dictionary.
        """
        search_config = super().get_search_config()
        search_config["initialize"] = self._initialize
        del search_config["verbose"]

        search_config = self._handle_gfo_defaults(search_config)

        search_config["search_space"] = self._to_dict_np(search_config["search_space"])

        return search_config

    def _handle_gfo_defaults(self, search_config):
        """Handle default values for GFO search configuration.

        Temporary measure until GFO handles defaults gracefully.

        Parameters
        ----------
        search_config : dict with str keys
            The search configuration dictionary to handle defaults for.

        Returns
        -------
        search_config : dict with str keys
            The search configuration dictionary with defaults handled.
        """
        if "sampling" in search_config and search_config["sampling"] is None:
            search_config["sampling"] = {"random": 1000000}

        if "tree_para" in search_config and search_config["tree_para"] is None:
            search_config["tree_para"] = {"n_estimators": 100}

        return search_config

    def _to_dict_np(self, search_space):
        """Coerce the search space to a format suitable for gfo optimizers.

        gfo expects dicts of numpy arrays, not lists. This method coerces
        lists or tuples in the search space to numpy arrays.

        This method also supports sklearn-style union grids (list of dicts),
        which are flattened into a single indexed search space.

        Parameters
        ----------
        search_space : dict or list of dict
            The search space to coerce. Can be:
            - dict with str keys and array-like values (standard grid)
            - list of dicts (sklearn-style union grid)

        Returns
        -------
        dict with str keys and 1D numpy arrays as values
            The coerced search space.
        """
        import numpy as np

        # Handle sklearn-style union grids (list of dicts)
        if isinstance(search_space, list):
            return self._handle_union_grid(search_space)

        def coerce_to_numpy(arr):
            """Coerce a list or tuple to a numpy array."""
            if not isinstance(arr, np.ndarray):
                return np.array(arr)
            return arr

        self._union_grid_configs = None  # Not a union grid
        coerced_search_space = {k: coerce_to_numpy(v) for k, v in search_space.items()}
        return coerced_search_space

    def _handle_union_grid(self, search_space):
        """Handle sklearn-style union grids (list of dicts).

        Union grids allow specifying different parameter combinations for
        different sub-grids. For example::

            [
                {'kernel': ['linear'], 'C': [0.1, 1, 10]},
                {'kernel': ['rbf'], 'C': [0.1, 1], 'gamma': [0.01, 0.1]},
            ]

        This is converted to an indexed search space where each index
        corresponds to a specific configuration from the union grid.

        Parameters
        ----------
        search_space : list of dict
            sklearn-style union grid.

        Returns
        -------
        dict
            Search space with a single ``_config_idx`` dimension containing
            indices into the flattened configuration list.
        """
        import numpy as np
        from sklearn.model_selection import ParameterGrid

        # Generate all configurations from the union grid
        all_configs = list(ParameterGrid(search_space))

        if not all_configs:
            raise ValueError("Union grid resulted in empty configuration set.")

        # Store configurations for decoding
        self._union_grid_configs = all_configs

        # Create a simple indexed search space
        # GFO will search over config indices
        return {"_config_idx": np.arange(len(all_configs))}

    def _decode_union_grid(self, params):
        """Decode union grid index back to actual parameters.

        Parameters
        ----------
        params : dict
            Parameter dict containing ``_config_idx``.

        Returns
        -------
        dict
            Actual parameter configuration from the union grid.
        """
        if self._union_grid_configs is None:
            return params

        config_idx = int(params.get("_config_idx", 0))
        if 0 <= config_idx < len(self._union_grid_configs):
            return self._union_grid_configs[config_idx].copy()
        return params

    def _solve(self, experiment, **search_config):
        """Run the optimization search process.

        Parameters
        ----------
        experiment : BaseExperiment
            The experiment to optimize parameters for.
        search_config : dict with str keys
            identical to return of ``get_search_config``.

        Returns
        -------
        dict with str keys
            The best parameters found during the search.
            Must have keys a subset or identical to experiment.paramnames().
        """
        n_iter = search_config.pop("n_iter", 100)
        max_time = search_config.pop("max_time", None)

        gfo_cls = self._get_gfo_class()
        gfopt = gfo_cls(**search_config)

        # Wrap objective to handle union grid decoding
        def _objective(params):
            decoded_params = self._decode_union_grid(params)
            score, _ = experiment.score(decoded_params)
            return score

        with StdoutMute(active=not self.verbose):
            gfopt.search(
                objective_function=_objective,
                n_iter=n_iter,
                max_time=max_time,
            )

        best_params = gfopt.best_para
        best_params = self._decode_union_grid(best_params)
        return best_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        ``get_test_params`` is a unified interface point to store
        parameter settings for testing purposes. This function is also
        used in ``create_test_instance`` and ``create_test_instances_and_names``
        to construct test instances.

        ``get_test_params`` should return a single ``dict``, or a ``list`` of ``dict``.

        Each ``dict`` is a parameter configuration for testing,
        and can be used to construct an "interesting" test instance.
        A call to ``cls(**params)`` should
        be valid for all dictionaries ``params`` in the return of ``get_test_params``.

        The ``get_test_params`` need not return fixed lists of dictionaries,
        it can also return dynamic or stochastic parameter settings.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        import numpy as np

        from hyperactive.experiment.integrations import SklearnCvExperiment

        sklearn_exp = SklearnCvExperiment.create_test_instance()
        params_sklearn = {
            "experiment": sklearn_exp,
            "search_space": {
                "C": np.array([0.01, 0.1, 1, 10]),
                "gamma": np.array([0.0001, 0.01, 0.1, 1, 10]),
            },
            "n_iter": 100,
        }

        from hyperactive.experiment.bench import Ackley

        ackley_exp = Ackley.create_test_instance()
        params_ackley = {
            "experiment": ackley_exp,
            "search_space": {
                "x0": np.linspace(-5, 5, 10),
                "x1": np.linspace(-5, 5, 10),
            },
            "n_iter": 100,
        }
        params_ackley_list = {
            "experiment": ackley_exp,
            "search_space": {
                "x0": list(np.linspace(-5, 5, 10)),
                "x1": list(np.linspace(-5, 5, 10)),
            },
            "n_iter": 100,
        }
        return [params_sklearn, params_ackley, params_ackley_list]
