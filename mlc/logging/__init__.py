import time
import datetime
from typing import Any

from comet_ml import Experiment

from mlc.logging.comet_config import (
    COMET_APIKEY,
    COMET_PROJECT,
    COMET_WORKSPACE,
)


class XP(object):
    def __init__(
        self,
        args: Any,
        project_name: str = COMET_PROJECT,
        workspace: str = COMET_WORKSPACE,
    ) -> None:
        if (project_name == "") or (project_name is None):
            self.xp = None
        else:
            timestamp = datetime.datetime.now()
            timestamp = f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S%z}'
            #timestamp = time.time()
            args["timestamp"] = timestamp
            self.experiment_name = "mcl_{}_{}_{}_{}".format(
                args.get("model_name", ""),
                args.get("dataset", ""),
                args.get("trial", ""),
                timestamp
            )
            experiment = Experiment(
                api_key=COMET_APIKEY,
                project_name=project_name,
                workspace=workspace,
                auto_param_logging=False,
                auto_metric_logging=False,
                parse_args=False,
                display_summary=False,
                disabled=False,
            )

            experiment.set_name(self.experiment_name)
            experiment.log_parameters(args)

            self.xp = experiment

    def log_parameters(self, *args: Any, **kwargs: Any) -> None:
        if self.xp is not None:
            self.xp.log_parameters(*args, **kwargs)
        else:
            print("logging parameters", *args)
            print("logging parameters", kwargs)

    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        if self.xp is not None:
            self.xp.log_metrics(*args, **kwargs)
        else:
            print("logging metrics", *args)
            print("logging metrics", kwargs)

    def log_metric(self, name: str, value: Any, **kwargs: Any) -> None:
        if self.xp is not None:
            self.xp.log_metric(name, value, **kwargs)
        else:
            print("logging metric", name, value)
            print("logging metric", kwargs)

    def end(self) -> None:
        if self.xp is not None:
            self.xp.end()

    def log_model(self, name: str, path: str) -> None:
        if self.xp is not None:
            self.xp.log_model(name, path, copy_to_tmp=False)
        else:
            print("Logging model", name, path)

    def log_table(self, *args: Any, **kwargs: Any) -> None:
        if self.xp is not None:
            self.xp.log_table(*args, **kwargs)
        else:
            print("logging table metrics", *args)
            print("logging table metrics", kwargs)

    def get_name(self) -> str:
        if self.xp is not None:
            return self.xp.get_name() + "_"
        else:
            return ""
