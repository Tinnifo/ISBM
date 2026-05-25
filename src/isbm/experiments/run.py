from contextlib import contextmanager, nullcontext
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from isbm.metrics.clustering import ari
from isbm.utils.seeding import set_global_seed


@contextmanager
def _mlflow_run(cfg: DictConfig):
    import mlflow

    uri = cfg.tracking.tracking_uri
    if uri.startswith("file:./") or uri.startswith("file:."):
        # Resolve relative MLflow store against the project root, not Hydra's run dir.
        rel = uri.replace("file:", "", 1)
        abs_path = (Path.cwd() / rel).resolve()
        uri = f"file:{abs_path}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)
    with mlflow.start_run() as run:
        flat = _flatten(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_params({k: v for k, v in flat.items() if not k.startswith("tracking.")})
        yield run


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            out.update(_flatten(v, key))
    else:
        out[prefix] = d
    return out


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))
    set_global_seed(cfg.seed)

    X, Z_true = instantiate(cfg.data)
    model = instantiate(cfg.model)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker = _mlflow_run(cfg) if cfg.tracking.enabled else nullcontext()
    with tracker:
        model.fit(X)

        final_ll = float(model.ll_trace_[-1])
        ari_score = ari(Z_true, model.z1_)
        n_clusters = int(model.z1_.max() + 1)

        artifact_path = run_dir / "artifacts.npz"
        np.savez(
            artifact_path,
            X=X,
            Z_true=Z_true,
            z1=model.z1_,
            z2=model.z2_,
            ll_trace=np.asarray(model.ll_trace_),
        )

        if cfg.tracking.enabled:
            import mlflow

            mlflow.log_metric("final_ll", final_ll)
            mlflow.log_metric("ari", ari_score)
            mlflow.log_metric("n_clusters_inferred", n_clusters)
            mlflow.log_artifact(str(artifact_path))

        print(f"final_ll={final_ll:.3f}  ari={ari_score:.3f}  n_clusters={n_clusters}")
        return final_ll


if __name__ == "__main__":
    main()
