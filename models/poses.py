import torch
import torch.nn as nn
from utils.lie_group_helper import make_c2w


class LearnPose__(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose__, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w

class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R=True, learn_t=True, init_c2w=None):
        """
        Learnable camera pose module.

        :param num_cams:    Total number of cameras (N)
        :param learn_R:     Whether to learn rotation
        :param learn_t:     Whether to learn translation
        :param init_c2w:    Optional initial poses, shape (N, 4, 4) tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams

        # Register non-learnable initial poses, if provided
        if init_c2w is not None:
            self.register_buffer("init_c2w", init_c2w)
        else:
            self.init_c2w = None

        # Fixed (non-learnable) pose for camera 0
        self.fixed_r = torch.zeros(3, dtype=torch.float32)
        self.fixed_t = torch.zeros(3, dtype=torch.float32)

        # Learnable poses for cameras 1 to N-1
        self.r = nn.Parameter(torch.zeros(size=(num_cams - 1, 3), dtype=torch.float32), requires_grad=learn_R)
        self.t = nn.Parameter(torch.zeros(size=(num_cams - 1, 3), dtype=torch.float32), requires_grad=learn_t)

    def forward(self, cam_id):
        """
        Get camera-to-world pose for a specific camera ID.

        :param cam_id: Integer camera index (0 <= cam_id < num_cams)
        :return: (4, 4) camera-to-world transformation matrix
        """
        if cam_id == 0:
            device = self.r.device
            r = self.fixed_r.to(device)  # axis-angle
            t = self.fixed_t.to(device)
        else:
            r = self.r[cam_id - 1]
            t = self.t[cam_id - 1]

        c2w = make_c2w(r, t)  # shape (4, 4)

        # If initial pose is provided, compose it
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w

