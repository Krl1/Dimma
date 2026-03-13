import atexit
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import torch.nn as nn
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.utilities import _scan_checkpoints
try:
    import wandb
except ImportError:
    wandb = None

from pytorch_lightning.loggers.wandb import (
    WandbLogger,
)

if wandb is not None:
    try:
        from wandb.sdk.wandb_run import Run
    except ImportError:
        Run = Any
    try:
        from wandb.sdk.lib.disabled import RunDisabled
    except ImportError:
        RunDisabled = Any
else:
    Run = Any
    RunDisabled = Any

_WANDB_GREATER_EQUAL_0_10_22 = True  # Assuming modern wandb (>= 0.10.22)
if wandb is not None:
    from distutils.version import LooseVersion
    _WANDB_GREATER_EQUAL_0_10_22 = LooseVersion(wandb.__version__) >= LooseVersion("0.10.22")
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from torch import Tensor


class _CfgCache:
    def __init__(self, owner: "WandbLoggerWithCache") -> None:
        self._owner = owner

    def update(self, params: Dict[str, Any], allow_val_change: bool = True) -> None:
        _ = allow_val_change
        self._owner._cfg_cache.update(params)

    def __setitem__(self, key: str, value: Any) -> None:
        self._owner._cfg_cache[key] = value


class _ExpCache:
    def __init__(self, owner: "WandbLoggerWithCache") -> None:
        self._owner = owner
        self._cfg = _CfgCache(owner)

    @property
    def config(self) -> _CfgCache:
        return self._cfg

    def log(self, metrics: Mapping[str, Any]) -> None:
        self._owner._evt_cache.append(("log", dict(metrics)))

    def log_artifact(self, artifact: Any, aliases: Optional[List[str]] = None) -> None:
        self._owner._evt_cache.append(("artifact", (artifact, aliases)))

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        self._owner._evt_cache.append(("metric_def", (args, kwargs)))

    def watch(self, model: nn.Module, log: str = "gradients", log_freq: int = 100, log_graph: bool = True) -> None:
        self._owner._evt_cache.append(("watch", (model, log, log_freq, log_graph)))

    def unwatch(self, model: Optional[nn.Module] = None) -> None:
        self._owner._evt_cache.append(("unwatch", model))

    def use_artifact(self, artifact: str, type: Optional[str] = None) -> Any:
        run = self._owner._run_start()
        return run.use_artifact(artifact, type=type)

    def finish(self) -> None:
        self._owner._flush_end()

    def __getattr__(self, name: str) -> Any:
        run = self._owner._run_start()
        return getattr(run, name)


class WandbLoggerWithCache(WandbLogger):
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union[Run, RunDisabled, None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if wandb is None:
            raise ModuleNotFoundError("You want to use `wandb` logger which is not installed yet, install it with `pip install wandb`.")

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        if log_model and not _WANDB_GREATER_EQUAL_0_10_22:
            rank_zero_warn(
                f"Providing log_model={log_model} requires wandb version >= 0.10.22"
                " for logging associated model metadata.\n"
                "Hint: Upgrade with `pip install --upgrade wandb`."
            )

        Logger.__init__(self)
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time: Dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None

        if save_dir is not None:
            save_dir = os.fspath(save_dir)
        elif dir is not None:
            dir = os.fspath(dir)

        project = project or os.environ.get("WANDB_PROJECT", "lightning_logs")
        self._wandb_init: Dict[str, Any] = {
            "name": name,
            "project": project,
            "dir": save_dir or dir,
            "id": version or id,
            "resume": "allow",
            "anonymous": ("allow" if anonymous else None),
        }
        self._wandb_init.update(**kwargs)
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._checkpoint_name = checkpoint_name

        self._evt_cache: List[tuple[str, Any]] = []
        self._cfg_cache: Dict[str, Any] = {}
        self._exp_cache = _ExpCache(self)
        self._is_done = False
        self._fin_status = "success"
        atexit.register(self._flush_end)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name
        state["_experiment"] = None
        state["_exp_cache"] = None
        return state

    @property
    def experiment(self) -> Union[Run, RunDisabled, _ExpCache]:
        if self._experiment is not None:
            return self._experiment
        return self._exp_cache

    def _run_start(self) -> Union[Run, RunDisabled]:
        if self._experiment is not None:
            return self._experiment

        if self._offline:
            os.environ["WANDB_MODE"] = "dryrun"

        attach_id = getattr(self, "_attach_id", None)
        if wandb.run is not None:
            rank_zero_warn(
                "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
            )
            self._experiment = wandb.run
        elif attach_id is not None and hasattr(wandb, "_attach"):
            self._experiment = wandb._attach(attach_id)
        else:
            self._experiment = wandb.init(**self._wandb_init)

            if isinstance(self._experiment, (Run, RunDisabled)) and getattr(self._experiment, "define_metric", None):
                self._experiment.define_metric("trainer/global_step")
                self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        self._cfg_cache.update(params)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        data = dict(_add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR))
        if step is not None:
            data["trainer/global_step"] = step
        self._evt_cache.append(("log", data))

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        metrics = {key: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_text(
        self,
        key: str,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[str]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        self.log_table(key, columns, data, dataframe, step)

    @rank_zero_only
    def log_image(self, key: str, images: List[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kw_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]
        metrics = {key: [wandb.Image(img, **kw) for img, kw in zip(images, kw_list)]}
        self.log_metrics(metrics, step)

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        return self._project

    @property
    def version(self) -> Optional[str]:
        return self._experiment.id if self._experiment else self._id

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self._fin_status = status

    def _flush(self) -> None:
        if getattr(rank_zero_only, "rank", 0) != 0:
            return

        if self._checkpoint_callback is not None:
            self._scan_and_log_checkpoints(self._checkpoint_callback)
            self._checkpoint_callback = None

        has_data = bool(self._cfg_cache) or bool(self._evt_cache)
        if not has_data and self._experiment is None:
            return

        run = self._run_start()
        if self._cfg_cache:
            run.config.update(self._cfg_cache, allow_val_change=True)

        for evt, payload in self._evt_cache:
            if evt == "log":
                run.log(payload)
            elif evt == "artifact":
                art, aliases = payload
                if aliases is None:
                    run.log_artifact(art)
                else:
                    run.log_artifact(art, aliases=aliases)
            elif evt == "metric_def":
                args, kwargs = payload
                run.define_metric(*args, **kwargs)
            elif evt == "watch":
                model, log, log_freq, log_graph = payload
                run.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)
            elif evt == "unwatch":
                run.unwatch(payload)

        self._evt_cache.clear()
        self._cfg_cache.clear()

    def _flush_end(self) -> None:
        if self._is_done:
            return

        self._flush()
        if self._experiment is not None:
            self._experiment.finish(exit_code=0 if self._fin_status == "success" else 1)
        self._is_done = True