"""
Load VPT and IDM models WITHOUT requiring MineRL/Java environment.

The original load_model.py calls gym.make("MineRLBasaltFindCave-v0") which
requires Java + MineRL installed. This version bypasses that by directly
constructing the agent with hardcoded action spaces (identical to MineRL's).
"""
import pickle
import torch as th
from gym3.types import DictType

from lib.action_mapping import CameraHierarchicalMapping, IDMActionMapping
from lib.actions import ActionTransformer
from lib.policy import MinecraftAgentPolicy, InverseActionPolicy
from lib.torch_util import set_default_torch_device


AGENT_RESOLUTION = (128, 128)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[128, 128, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

import numpy as np
import cv2


class VPTAgent:
    """VPT agent that loads without MineRL environment."""

    def __init__(self, device='cuda', policy_kwargs=None, pi_head_kwargs=None):
        self.device = th.device(device)
        set_default_torch_device(self.device)

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(
            policy_kwargs=policy_kwargs,
            pi_head_kwargs=pi_head_kwargs,
            action_space=action_space,
        )

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(self.device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(self.device)

    def load_weights(self, path):
        self.policy.load_state_dict(
            th.load(path, map_location=self.device), strict=False)
        self.reset()

    def reset(self):
        self.hidden_state = self.policy.initial_state(1)

    def _env_obs_to_agent(self, minerl_obs):
        agent_input = cv2.resize(
            minerl_obs["pov"], AGENT_RESOLUTION,
            interpolation=cv2.INTER_LINEAR)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        return agent_input

    def _env_action_to_agent(self, minerl_action_transformed,
                              to_torch=False, check_if_null=False):
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if (np.all(minerl_action["buttons"] == 0) and
                np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin)):
                return None
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: th.from_numpy(v).to(self.device) for k, v in action.items()}
        return action


class IDMAgentNoJava:
    """IDM agent that loads without MineRL environment."""

    def __init__(self, idm_net_kwargs, pi_head_kwargs, device='cuda'):
        self.device = th.device(device)
        set_default_torch_device(self.device)

        self.action_mapper = IDMActionMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        idm_policy_kwargs = dict(
            idm_net_kwargs=idm_net_kwargs,
            pi_head_kwargs=pi_head_kwargs,
            action_space=action_space,
        )

        self.policy = InverseActionPolicy(**idm_policy_kwargs).to(self.device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(self.device)

    def load_weights(self, path):
        self.policy.load_state_dict(
            th.load(path, map_location=self.device), strict=False)
        self.reset()

    def reset(self):
        self.hidden_state = self.policy.initial_state(1)

    def _video_obs_to_agent(self, video_frames):
        imgs = [cv2.resize(f, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
                for f in video_frames]
        imgs = np.stack(imgs)[None]
        agent_input = {"img": th.from_numpy(imgs).to(self.device)}
        return agent_input

    def _agent_action_to_env(self, agent_action):
        action = {
            "buttons": agent_action["buttons"].cpu().numpy(),
            "camera": agent_action["camera"].cpu().numpy(),
        }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    def predict_actions(self, video_frames):
        agent_input = self._video_obs_to_agent(video_frames)
        dummy_first = th.zeros((video_frames.shape[0], 1)).to(self.device)
        predicted_actions, self.hidden_state, _ = self.policy.predict(
            agent_input, first=dummy_first, state_in=self.hidden_state,
            deterministic=True)
        predicted_minerl_action = self._agent_action_to_env(predicted_actions)
        return predicted_minerl_action


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def load_vpt_model(model_path: str, weights_path: str,
                   device: str = 'cuda') -> VPTAgent:
    """Load the VPT foundation model (no Java/MineRL needed)."""
    policy_kwargs, pi_head_kwargs = load_model_parameters(model_path)
    agent = VPTAgent(device=device, policy_kwargs=policy_kwargs,
                     pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights_path)
    return agent


def load_idm_model(model_path: str, weights_path: str,
                   device: str = 'cuda') -> IDMAgentNoJava:
    """Load the IDM model (no Java/MineRL needed)."""
    agent_parameters = pickle.load(open(model_path, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgentNoJava(idm_net_kwargs=net_kwargs,
                           pi_head_kwargs=pi_head_kwargs, device=device)
    agent.load_weights(weights_path)
    return agent
