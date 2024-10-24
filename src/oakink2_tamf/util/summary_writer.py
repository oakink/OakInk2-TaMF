from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from .ddp_util import master_only


class DDPSummaryWriter(SummaryWriter):

    def __init__(self, log_dir, rank, **kwargs):
        self.rank = rank
        self.ddp_log_dir = log_dir
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            assert self.rank == rank

        if not self.rank and self.ddp_log_dir is not None:
            self.ddp_enabled = True
            super(DDPSummaryWriter, self).__init__(self.ddp_log_dir, **kwargs)
        else:
            self.ddp_enabled = False

    @master_only
    def add_scalar(self, tag, value, global_step=None, walltime=None):
        if self.ddp_enabled:
            super(DDPSummaryWriter, self).add_scalar(tag, value, global_step=global_step, walltime=walltime)

    @master_only
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.ddp_enabled:
            return super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    @master_only
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"):
        if self.ddp_enabled:
            return super().add_image(tag, img_tensor, global_step, walltime=walltime, dataformats=dataformats)

    @master_only
    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats="NCHW"):
        if self.ddp_enabled:
            return super().add_images(tag, img_tensor, global_step, walltime=walltime, dataformats=dataformats)

    # there are lots of things can be added to tensorboard
