from .sac import SAC
from .sacfd import SACfD
from .sac_drq import SAC_DrQ

RL_ALGORITHMS = {
    SAC.name: SAC,
    SACfD.name: SACfD,
    SAC_DrQ.name: SAC_DrQ
}
