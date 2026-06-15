from lightning.fabric.plugins.environments.lightning import LightningEnvironment


class LocalEnvironment(LightningEnvironment):
    """A thin wrapper around Lightning's default LightningEnvironment.

    It is kept as an extension point so that custom multi-node launchers
    (SLURM, K8s, etc.) can be plugged in by overriding the relevant
    properties (e.g. ``main_address``, ``main_port``, ``world_size``,
    ``global_rank``, ``node_rank``).
    """
    pass
