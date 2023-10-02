import json
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datasets import IterableDataset
from huggingface_hub import EvalResult, HfApi, ModelCard, ModelCardData
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from transformers import PreTrainedModel, ProcessorMixin


PRETTY_TASK_NAMES = {
    "atari-alien": "ALE/Alien-v5",
    "atari-amidar": "ALE/Amidar-v5",
    "atari-assault": "ALE/Assault-v5",
    "atari-asterix": "ALE/Asterix-v5",
    "atari-asteroids": "ALE/Asteroids-v5",
    "atari-atlantis": "ALE/Atlantis-v5",
    "atari-bankheist": "ALE/BankHeist-v5",
    "atari-battlezone": "ALE/BattleZone-v5",
    "atari-beamrider": "ALE/BeamRider-v5",
    "atari-berzerk": "ALE/Berzerk-v5",
    "atari-bowling": "ALE/Bowling-v5",
    "atari-boxing": "ALE/Boxing-v5",
    "atari-breakout": "ALE/Breakout-v5",
    "atari-centipede": "ALE/Centipede-v5",
    "atari-choppercommand": "ALE/ChopperCommand-v5",
    "atari-crazyclimber": "ALE/CrazyClimber-v5",
    "atari-defender": "ALE/Defender-v5",
    "atari-demonattack": "ALE/DemonAttack-v5",
    "atari-doubledunk": "ALE/DoubleDunk-v5",
    "atari-enduro": "ALE/Enduro-v5",
    "atari-fishingderby": "ALE/FishingDerby-v5",
    "atari-freeway": "ALE/Freeway-v5",
    "atari-frostbite": "ALE/Frostbite-v5",
    "atari-gopher": "ALE/Gopher-v5",
    "atari-gravitar": "ALE/Gravitar-v5",
    "atari-hero": "ALE/Hero-v5",
    "atari-icehockey": "ALE/IceHockey-v5",
    "atari-jamesbond": "ALE/Jamesbond-v5",
    "atari-kangaroo": "ALE/Kangaroo-v5",
    "atari-krull": "ALE/Krull-v5",
    "atari-kungfumaster": "ALE/KungFuMaster-v5",
    "atari-montezumarevenge": "ALE/MontezumaRevenge-v5",
    "atari-mspacman": "ALE/MsPacman-v5",
    "atari-namethisgame": "ALE/NameThisGame-v5",
    "atari-phoenix": "ALE/Phoenix-v5",
    "atari-pitfall": "ALE/Pitfall-v5",
    "atari-pong": "ALE/Pong-v5",
    "atari-privateeye": "ALE/PrivateEye-v5",
    "atari-qbert": "ALE/Qbert-v5",
    "atari-riverraid": "ALE/Riverraid-v5",
    "atari-roadrunner": "ALE/RoadRunner-v5",
    "atari-robotank": "ALE/Robotank-v5",
    "atari-seaquest": "ALE/Seaquest-v5",
    "atari-skiing": "ALE/Skiing-v5",
    "atari-solaris": "ALE/Solaris-v5",
    "atari-spaceinvaders": "ALE/SpaceInvaders-v5",
    "atari-stargunner": "ALE/StarGunner-v5",
    "atari-surround": "ALE/Surround-v5",
    "atari-tennis": "ALE/Tennis-v5",
    "atari-timepilot": "ALE/TimePilot-v5",
    "atari-tutankham": "ALE/Tutankham-v5",
    "atari-upndown": "ALE/UpNDown-v5",
    "atari-venture": "ALE/Venture-v5",
    "atari-videopinball": "ALE/VideoPinball-v5",
    "atari-wizardofwor": "ALE/WizardOfWor-v5",
    "atari-yarsrevenge": "ALE/YarsRevenge-v5",
    "atari-zaxxon": "ALE/Zaxxon-v5",
    "babyai-action-obj-door": "BabyAI-ActionObjDoor-v0",
    "babyai-blocked-unlock-pickup": "BabyAI-BlockedUnlockPickup-v0",
    "babyai-boss-level-no-unlock": "BabyAI-BossLevelNoUnlock-v0",
    "babyai-boss-level": "BabyAI-BossLevel-v0",
    "babyai-find-obj-s5": "BabyAI-FindObjS5-v0",
    "babyai-go-to-door": "BabyAI-GoToDoor-v0",
    "babyai-go-to-imp-unlock": "BabyAI-GoToImpUnlock-v0",
    "babyai-go-to-local": "BabyAI-GoToLocal-v0",
    "babyai-go-to-obj-door": "BabyAI-GoToObjDoor-v0",
    "babyai-go-to-obj": "BabyAI-GoToObj-v0",
    "babyai-go-to-red-ball-grey": "BabyAI-GoToRedBallGrey-v0",
    "babyai-go-to-red-ball-no-dists": "BabyAI-GoToRedBallNoDists-v0",
    "babyai-go-to-red-ball": "BabyAI-GoToRedBall-v0",
    "babyai-go-to-red-blue-ball": "BabyAI-GoToRedBlueBall-v0",
    "babyai-go-to-seq": "BabyAI-GoToSeq-v0",
    "babyai-go-to": "BabyAI-GoTo-v0",
    "babyai-key-corridor": "BabyAI-KeyCorridor-v0",
    "babyai-key-in-box": "BabyAI-KeyInBox-v0",
    "babyai-mini-boss-level": "BabyAI-MiniBossLevel-v0",
    "babyai-move-two-across": "BabyAI-MoveTwoAcrossS8N9-v0",
    "babyai-one-room-s8": "BabyAI-OneRoomS8-v0",
    "babyai-open-door": "BabyAI-OpenDoor-v0",
    "babyai-open-doors-order": "BabyAI-OpenDoorsOrderN4-v0",
    "babyai-open-red-door": "BabyAI-OpenRedDoor-v0",
    "babyai-open-two-doors": "BabyAI-OpenTwoDoors-v0",
    "babyai-open": "BabyAI-Open-v0",
    "babyai-pickup-above": "BabyAI-PickupAbove-v0",
    "babyai-pickup-dist": "BabyAI-PickupDist-v0",
    "babyai-pickup-loc": "BabyAI-PickupLoc-v0",
    "babyai-pickup": "BabyAI-Pickup-v0",
    "babyai-synth-loc": "BabyAI-SynthLoc-v0",
    "babyai-synth-seq": "BabyAI-SynthSeq-v0",
    "babyai-synth": "BabyAI-Synth-v0",
    "babyai-unblock-pickup": "BabyAI-UnblockPickup-v0",
    "babyai-unlock-local": "BabyAI-UnlockLocal-v0",
    "babyai-unlock-pickup": "BabyAI-UnlockPickup-v0",
    "babyai-unlock-to-unlock": "BabyAI-UnlockToUnlock-v0",
    "babyai-unlock": "BabyAI-Unlock-v0",
    "conceptual-captions": "Conceptual Captions",
    "metaworld-assembly": "assembly-v2",
    "metaworld-basketball": "basketball-v2",
    "metaworld-bin-picking": "bin-picking-v2",
    "metaworld-box-close": "box-close-v2",
    "metaworld-button-press-topdown-wall": "button-press-topdown-wall-v2",
    "metaworld-button-press-topdown": "button-press-topdown-v2",
    "metaworld-button-press-wall": "button-press-wall-v2",
    "metaworld-button-press": "button-press-v2",
    "metaworld-coffee-button": "coffee-button-v2",
    "metaworld-coffee-pull": "coffee-pull-v2",
    "metaworld-coffee-push": "coffee-push-v2",
    "metaworld-dial-turn": "dial-turn-v2",
    "metaworld-disassemble": "disassemble-v2",
    "metaworld-door-close": "door-close-v2",
    "metaworld-door-lock": "door-lock-v2",
    "metaworld-door-open": "door-open-v2",
    "metaworld-door-unlock": "door-unlock-v2",
    "metaworld-drawer-close": "drawer-close-v2",
    "metaworld-drawer-open": "drawer-open-v2",
    "metaworld-faucet-close": "faucet-close-v2",
    "metaworld-faucet-open": "faucet-open-v2",
    "metaworld-hammer": "hammer-v2",
    "metaworld-hand-insert": "hand-insert-v2",
    "metaworld-handle-press-side": "handle-press-side-v2",
    "metaworld-handle-press": "handle-press-v2",
    "metaworld-handle-pull-side": "handle-pull-side-v2",
    "metaworld-handle-pull": "handle-pull-v2",
    "metaworld-lever-pull": "lever-pull-v2",
    "metaworld-peg-insert-side": "peg-insert-side-v2",
    "metaworld-peg-unplug-side": "peg-unplug-side-v2",
    "metaworld-pick-out-of-hole": "pick-out-of-hole-v2",
    "metaworld-pick-place-wall": "pick-place-wall-v2",
    "metaworld-pick-place": "pick-place-v2",
    "metaworld-plate-slide-back-side": "plate-slide-back-side-v2",
    "metaworld-plate-slide-back": "plate-slide-back-v2",
    "metaworld-plate-slide-side": "plate-slide-side-v2",
    "metaworld-plate-slide": "plate-slide-v2",
    "metaworld-push-back": "push-back-v2",
    "metaworld-push-wall": "push-wall-v2",
    "metaworld-push": "push-v2",
    "metaworld-reach-wall": "reach-wall-v2",
    "metaworld-reach": "reach-v2",
    "metaworld-shelf-place": "shelf-place-v2",
    "metaworld-soccer": "soccer-v2",
    "metaworld-stick-pull": "stick-pull-v2",
    "metaworld-stick-push": "stick-push-v2",
    "metaworld-sweep-into": "sweep-into-v2",
    "metaworld-sweep": "sweep-v2",
    "metaworld-window-close": "window-close-v2",
    "metaworld-window-open": "window-open-v2",
    "mujoco-ant": "Ant-v4",
    "mujoco-doublependulum": "InvertedDoublePendulum-v4",
    "mujoco-halfcheetah": "HalfCheetah-v4",
    "mujoco-hopper": "Hopper-v4",
    "mujoco-humanoid": "Humanoid-v4",
    "mujoco-pendulum": "InvertedPendulum-v4",
    "mujoco-pusher": "Pusher-v4",
    "mujoco-reacher": "Reacher-v4",
    "mujoco-standup": "HumanoidStandup-v4",
    "mujoco-swimmer": "Swimmer-v4",
    "mujoco-walker": "Walker2d-v4",
    "ok-vqa": "OK-VQA",
    "oscar": "OSCAR",
}


@contextmanager
def suppress_stdout():
    class DummyFile(object):
        def write(self, x):
            pass

    # Save the current stdout
    original_stdout = sys.stdout
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stdout = original_stdout


def no_print_decorator(func):
    def wrapper(*args, **kwargs):
        with suppress_stdout():
            return func(*args, **kwargs)

    return wrapper


def compute_mse_loss(
    predicted: torch.FloatTensor, true: torch.FloatTensor, mask: torch.BoolTensor, weights: torch.FloatTensor = None
) -> torch.FloatTensor:
    """
    Compute the Mean Squared Error (MSE) loss between predicted and true observations, considering valid timesteps.

    Args:
        predicted (`torch.FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Predicted observations at the output of the model.
        true (`torch.FloatTensor` of shape `(batch_size, max_seq_len, ...)`):
            Ground truth observations.
        mask (`torch.BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.

    Returns:
        loss (`torch.FloatTensor` of shape `(,)`):
            MSE loss between predicted and true observations.
    """
    # Compute element-wise MSE loss
    loss = F.mse_loss(predicted, true, reduction="none")

    # Average the loss over all dimensions after the second one
    for dim in reversed(range(2, loss.dim())):
        loss = loss.mean(dim=dim)

    # Use the mask to zero out invalid entries
    loss = torch.sum(loss * mask, dim=1)

    # Apply weights if provided
    if weights is not None:
        loss = loss * weights

    # Sum the loss and normalize by the number of valid elements
    loss = loss.sum() / mask.sum()

    return loss


def compute_ce_loss(
    predicted: torch.FloatTensor, true: torch.LongTensor, mask: torch.BoolTensor, weights: torch.FloatTensor = None
) -> torch.FloatTensor:
    """
    Compute the Cross Entropy (CE) loss between predicted logits and true class labels, considering valid timesteps.

    Args:
        predicted (`torch.FloatTensor` of shape `(batch_size, max_seq_len, num_classes)`):
            Predicted logits at the output of the model.
        true (`torch.LongTensor` of shape `(batch_size, max_seq_len)`):
            Ground truth class labels.
        mask (`torch.BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.

    Returns:
        loss (`torch.FloatTensor` of shape `(,)`):
            CE loss between predicted logits and true class labels.
    """

    # Compute element-wise CE loss
    loss = F.cross_entropy(predicted.view(-1, predicted.size(-1)), true.view(-1), reduction="none")
    loss = loss.view(true.size())

    # Use the mask to zero out invalid entries
    loss = torch.sum(loss * mask, dim=1)

    # Apply weights if provided
    if weights is not None:
        loss = loss * weights

    # Sum the loss and normalize by the number of valid elements
    loss = loss.sum() / mask.sum()

    return loss


def filter_tensor(
    tensor: Tensor, mask: Optional[BoolTensor] = None, sizes: Optional[LongTensor] = None
) -> List[List[Any]]:
    """
    Filters a tensor based on a mask and sizes, and returns a nested list of values.

    Args:
        tensor (`torch.Tensor` of shape `(batch_size, seq_len, ...)`):
            Input tensor to be filtered.
        mask (`Optional[torch.BoolTensor]` of shape `(batch_size, seq_len)`, **optional**):
            Boolean mask indicating valid timesteps. If None, all timesteps are considered valid.
        sizes (`Optional[torch.LongTensor]` of shape `(batch_size,)`, **optional**):
            Observation size for each example in the batch. If None, all sizes are considered valid.

    Returns:
        `List[List[Any]]`:
            A nested list containing filtered values, considering only valid timesteps and sizes.

    Examples:
        >>> tensor = torch.arange(12).reshape(2, 3, 2)
        >>> mask = torch.tensor([[True, True, False], [True, False, False]])
        >>> filter_tensor(tensor, mask)
        [[[0, 1], [2, 3]], [[6, 7]]]
        >>> sizes = torch.tensor([2, 1])
        >>> filter_tensor(tensor, sizes=sizes)
        [[[0, 1], [2, 3], [4, 5]], [[6], [8], [10]]]
    """
    batch_size, seq_len = tensor.shape[:2]
    nested_list = []

    for i in range(batch_size):
        batch_list = []
        for j in range(seq_len):
            if mask is None or mask[i, j].item() == 1:
                obs_size = sizes[i].item() if sizes is not None else tensor.shape[-1]
                values = tensor[i, j, :obs_size].tolist()
                batch_list.append(values)
        nested_list.append(batch_list)

    return nested_list


def cyclic_expand_dim(tensor: Tensor, expanded_dim_size: int) -> Tensor:
    """
    Expands the last dimension of a tensor cyclically to a specified size.

    Args:
        tensor (`torch.Tensor` of shape `(batch_size, seq_len, ...)`):
            Input tensor whose last dimension is to be expanded cyclically.
        expanded_dim_size (`int`):
            The desired size of the last dimension after expansion.

    Returns:
        `torch.Tensor` of shape `(batch_size, seq_len, expanded_dim_size)`:
            A tensor with its last dimension expanded cyclically to the specified size.

    Examples:
        >>> tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> cyclic_expand_dim(tensor, 5)
        tensor([[[1, 2, 1, 2, 1], [3, 4, 3, 4, 3]], [[5, 6, 5, 6, 5], [7, 8, 7, 8, 7]]])
    """
    B, L, X = tensor.shape
    if expanded_dim_size < X:
        raise ValueError(
            f"Expanded dimension size ({expanded_dim_size}) must be greater than the original dimension size ({X})."
        )
    indices = torch.arange(expanded_dim_size) % X
    return tensor[..., indices]


def write_video(frames: List[np.ndarray], filename: str, fps: int):
    """
    Writes a list of frames into a video file.

    Args:
        frames (`List[np.ndarray]`):
            List of frames in RGB format.
        filename (`str`):
            Output video filename including the extension.
        fps (`int`):
            Frames per second for the output video.
    """
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    shape = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(filename, fourcc, fps, shape)

    # Write frames to video
    for frame in frames:
        out.write(frame[..., [2, 1, 0]])  # convert RGB to BGR and write

    # Release resources
    out.release()


def stack_input(input_list: List[Any]) -> Tensor:
    # If the first element is a torch tensor
    if isinstance(input_list[0], torch.Tensor):
        return torch.stack(input_list)

    # If the first element is a numpy array
    elif isinstance(input_list[0], np.ndarray):
        return torch.stack([torch.tensor(item) for item in input_list])

    # If the input_list is a list of values (like Python lists or scalars)
    else:
        return torch.tensor(input_list)


def get_inner_shape(elmt):
    # If the first element is a torch tensor
    if isinstance(elmt, torch.Tensor):
        return elmt.shape

    # If the first element is a numpy array
    elif isinstance(elmt, np.ndarray):
        return elmt.shape

    # If the elmt is a list of values (like Python lists or scalars)
    else:
        # If the first element is a list or a scalar
        if isinstance(elmt, (list, int, float)):
            return (len(elmt),) if isinstance(elmt, list) else ()
        else:
            raise ValueError(f"Unsupported data type: {type(elmt)}")


def _collate(sequences: List[List], dtype: torch.dtype) -> Tuple[FloatTensor, BoolTensor]:
    """
    Collates a list of vectors into a single tensor.

    Args:
        sequences (`List[List]`):
            List of sequences, i.e. list of object that can be either single value or nested structure.

    Returns:
        collated (`torch.Tensor`):
            Collated tensor of shape `(batch_size, max_seq_len, ...)`.
        mask (`torch.BoolTensor` of shape `(batch_size, max_seq_len)`):
            Boolean mask indicating valid timesteps.
    """
    batch_size = len(sequences)
    max_seq_len = max([len(sequence) for sequence in sequences])

    data_shape = get_inner_shape(sequences[0][0])
    collated = torch.zeros(batch_size, max_seq_len, *data_shape, dtype=dtype)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    # Pad sequences with zeros
    for i, sequence in enumerate(sequences):
        seq_len = min(len(sequence), max_seq_len)
        collated[i, :seq_len] = stack_input(sequence[:seq_len])
        mask[i, :seq_len] = 1

    return collated, mask


def collate_fn(batch: List[Dict[str, List]]) -> Dict[str, Tensor]:
    collated = {}

    if "input_ids" in batch[0] and batch[0]["input_ids"] is not None:
        collated["input_ids"] = torch.tensor([x["input_ids"] for x in batch], dtype=torch.int64)
        collated["attention_mask"] = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.bool)

    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        collated["pixel_values"] = torch.stack([torch.from_numpy(x["pixel_values"]) for x in batch])

    continuous_keys = ["continuous_observations", "continuous_actions", "rewards"]
    for key in continuous_keys:
        if key in batch[0] and batch[0][key] is not None:
            values = [x[key] for x in batch]
            collated[key], collated["attention_mask"] = _collate(values, dtype=torch.float32)

    discrete_keys = ["discrete_observations", "discrete_actions", "text_observations"]
    for key in discrete_keys:
        if key in batch[0] and batch[0][key] is not None:
            values = [x[key] for x in batch]
            collated[key], _ = _collate(values, dtype=torch.int64)

    key = "image_observations"
    if key in batch[0] and batch[0][key] is not None:
        values = [x[key] for x in batch]
        collated[key], _ = _collate(values, dtype=torch.float32)

    if "loss_weight" in batch[0]:
        collated["loss_weight"] = torch.tensor([x["loss_weight"] for x in batch], dtype=torch.float32)

    return collated


def preprocess_function(examples: Dict[str, Any], max_len: int) -> Dict[str, Any]:
    """
    Splits the sequences in the input dictionary into chunks of a specified maximum length.

    Args:
        examples (dict of lists of lists):
            A dictionary where each key corresponds to a list of sequences. Each sequence is a list of elements.

    Returns:
        dict:
            A dictionary with the same keys as the input. Each key corresponds to a list of chunks, where each chunk
            is a subsequence of the input sequences with a length not exceeding the specified maximum length.

    Examples:
        >>> examples = {
        ...     "key1": [[1, 2, 3], [4, 5, 6, 7]],
        ...     "key2": [[8, 9, 10], [11, 12, 13, 14]}
        >>> preprocess_function(examples, max_len=2)
        {
            "key1": [[1, 2], [3], [4, 5], [6, 7]],
            "key2": [[8, 9], [10], [11, 12], [13, 14]],
        }
    """
    out_dict = {key: [] for key in examples.keys()}
    first_ep_batch = next(iter(examples.values()))
    num_episodes = len(first_ep_batch)
    for ep_idx in range(num_episodes):
        ep_len = len(first_ep_batch[ep_idx])
        for t in range(0, ep_len, max_len):
            for key in examples.keys():
                if hasattr(examples[key][ep_idx], "__len__"):
                    chunk = examples[key][ep_idx][t : t + max_len]
                else:
                    chunk = examples[key][ep_idx]
                out_dict[key].append(chunk)

    return out_dict


def generate_rl_eval_results(scores_dict: Dict[str, List[float]]) -> List[EvalResult]:
    """
    Generate a list of EvalResult objects.

    Args:
        scores_dict (`Dict[str, List[float]]`):
            Dictionary containing the scores for each task.

    Returns:
        `List[EvalResult]`:
            A list of EvalResult objects.
    """
    eval_results = []
    for task_name, scores in scores_dict.items():
        mean_reward = np.mean(scores)
        std_reward = np.std(scores)

        eval_results.append(
            EvalResult(
                task_type="reinforcement-learning",
                task_name="Reinforcement Learning",
                dataset_type=task_name,
                dataset_name=PRETTY_TASK_NAMES[task_name],
                metric_type="total_reward",
                metric_name="Total reward",
                metric_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            )
        )

    for task_name, scores in scores_dict.items():
        mean_reward = np.mean(scores)
        std_reward = np.std(scores)
        with open("gia/eval/rl/scores_dict.json", "r") as file:
            scores_dict = json.load(file)

        expert_score = scores_dict[task_name]["expert"]["mean"]
        random_score = scores_dict[task_name]["random"]["mean"]
        norm_mean_reward = (mean_reward - random_score) / (expert_score - random_score)
        norm_std_reward = std_reward / (expert_score - random_score)

        eval_results.append(
            EvalResult(
                task_type="reinforcement-learning",
                task_name="Reinforcement Learning",
                dataset_type=task_name,
                dataset_name=PRETTY_TASK_NAMES[task_name],
                metric_type="expert_normalized_total_reward",
                metric_name="Expert normalized total reward",
                metric_value=f"{norm_mean_reward:.2f} +/- {norm_std_reward:.2f}",
            )
        )
    return eval_results


def generate_model_card(model_name: str, scores_dict: Optional[Dict[str, List[float]]] = None) -> ModelCard:
    """
    Generate a ModelCard from a template.

    Args:
        model_name (`str`):
            Model name.
        scores_dict (`Dict[str, List[float]]`):
            Dictionary containing the scores for each task.

    Returns:
        `ModelCard`:
            A ModelCard object.
    """
    tags = ["reinforcement-learning"]
    if scores_dict is not None:
        tags.extend(scores_dict.keys())
    card_data = ModelCardData(
        tags=tags,
        eval_results=generate_rl_eval_results(scores_dict) if scores_dict is not None else None,
        model_name=model_name,
        datasets="gia-project/gia-dataset-parquet",
        pipeline_tag="reinforcement-learning",
    )
    card = ModelCard.from_template(
        card_data,
        template_path="templates/model_card.md",
        model_name=model_name,
        model_id="Gia2",
        tasks=[PRETTY_TASK_NAMES[task_name] for task_name in scores_dict.keys()] if scores_dict is not None else [],
    )
    return card


def push_to_hub(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    repo_id: str,
    scores_dict: Optional[Dict[str, List[float]]] = None,
    replay_path: Optional[str] = None,
) -> None:
    """
    Push a model to the Hugging Face Hub.

    Args:
        model (`PreTrainedModel`):
            Model to push.
        processor (`ProcessorMixin`):
            Processor to push.
        repo_id (`str`):
            Repository ID to push to.
        scores_dict (`Dict[str, List[float]]` or `None`, **optional**):
            Dictionary containing the scores for each task.
        replay_path (`str` or `None`, **optional**):
            Path to the replay video.
    """
    api = HfApi()

    # Create the repo
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Create a README.md using a template
    model_card = generate_model_card(repo_id, scores_dict)
    model_card.push_to_hub(repo_id, commit_message="Upload model card")

    # Push the model
    model.push_to_hub(repo_id, commit_message="Upload model")

    # Push the processor
    processor.push_to_hub(repo_id, commit_message="Upload processor")

    # Push the replay
    api.upload_file(
        path_or_fileobj=replay_path,
        path_in_repo="replay.mp4",
        repo_id=repo_id,
        commit_message="Upload replay",
        repo_type="model",
    )

    print(f"Pushed model to https://huggingface.co/{repo_id}")


def save_video_grid(
    videos: List[List[np.ndarray]],
    input_fps: List[int],
    output_filename: str = "output.mp4",
    width: int = 640,
    output_fps: int = 30,
    max_length_seconds: Optional[int] = None,
) -> None:
    """
    Save a grid video from a list of videos.

    Args:
        videos (`List[List[np.ndarray]]`):
            List of videos, where each video is a list of frames in RGB format.
        input_fps (`List[int]`):
            List of FPS values for each video.
        output_filename (`str`, **optional**):
            Output video filename including the extension.
        output_fps (`int`, **optional**):
            Frames per second for the output video.
        max_length_seconds (`Optional[int]`, **optional**):
            Maximum length of the output video in seconds. If None, the length of the longest video is used.
    """
    # Check if there are any videos
    if not videos:
        raise ValueError("No videos provided")

    if len(videos) != len(input_fps):
        raise ValueError("The number of videos must match the number of FPS values")

    # Determine grid size based on the number of videos
    num_cols = int(np.ceil(np.sqrt(len(videos))))
    num_rows = int(np.ceil(len(videos) / num_cols))
    height = width * num_rows // num_cols

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_filename = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(temp_filename, fourcc, output_fps, (width, height))

    # Number of frames in the longest video, if max_length_seconds is specified, adjust max_frames
    max_frames = max(len(video) for video in videos)
    if max_length_seconds is not None:
        max_frames = min(max_frames, output_fps * max_length_seconds)

    for frame_idx in range(max_frames):
        # Create an empty grid
        grid = np.zeros((height, width, 3), dtype=np.uint8)

        for video_idx, video in enumerate(videos):
            # Adjust for different FPS values
            adjusted_frame_idx = int((frame_idx * input_fps[video_idx]) / output_fps)
            looped_frame_idx = adjusted_frame_idx % len(video)
            frame = video[looped_frame_idx]
            row = video_idx // num_cols
            col = video_idx % num_cols
            # resize the frame to the grid size
            w = width // num_cols
            h = height // num_rows
            frame = cv2.resize(frame, (w, h))
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame

        grid = grid[..., [2, 1, 0]]  # RGB to BGR
        out.write(grid)

    out.release()
    os.system(f"ffmpeg -y -i {temp_filename} -vcodec h264 {output_filename}")


def mix_iterable_datasets(
    datasets: List[IterableDataset], batch_size: int, stopping_strategy: str = "first_exhausted"
):
    """
    Mixes multiple IterableDataset objects into a single IterableDataset.

    Args:
        datasets (`List[IterableDataset]`):
            List of IterableDataset objects.
        batch_size (`int`):
            Batch size.
        stopping_strategy (`str`, **optional**):
            Stopping strategy. Can be either "first_exhausted" or "all_exhausted".

    Returns:
        `IterableDataset`:
            A mixed IterableDataset object.
    """

    def generator(datasets: List[IterableDataset], batch_size: int, stopping_strategy: str = "first_exhausted"):
        assert stopping_strategy in ["first_exhausted", "all_exhausted"]
        iterators = [iter(dataset) for dataset in datasets]
        exhausted = [False] * len(datasets)  # A list to keep track of which iterators are exhausted

        while True:
            for i, it in enumerate(iterators):
                for _ in range(batch_size):
                    try:
                        yield next(it)
                    except StopIteration:
                        if stopping_strategy == "first_exhausted":
                            return
                        else:
                            # Mark the iterator as exhausted
                            exhausted[i] = True
                            # Check if all iterators are exhausted
                            if all(exhausted):
                                return
                            # Reinitialize the exhausted iterator
                            iterators[i] = iter(datasets[i])
                            try:
                                yield next(iterators[i])
                            except StopIteration:
                                # Handle the case when the reinitialized iterator is also exhausted immediately
                                return

    gen_kwargs = {"datasets": datasets, "batch_size": batch_size, "stopping_strategy": stopping_strategy}
    return IterableDataset.from_generator(generator=generator, gen_kwargs=gen_kwargs)
