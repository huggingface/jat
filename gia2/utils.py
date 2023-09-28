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
from transformers import AutoTokenizer, PreTrainedModel

from gia.eval.rl.envs.core import TASK_NAME_TO_ENV_ID


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


def generate_eval_results(scores_dict: Dict[str, List[float]]) -> List[EvalResult]:
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
                dataset_name=TASK_NAME_TO_ENV_ID[task_name],
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
                dataset_name=TASK_NAME_TO_ENV_ID[task_name],
                metric_type="expert_normalized_total_reward",
                metric_name="Expert normalized total reward",
                metric_value=f"{norm_mean_reward:.2f} +/- {norm_std_reward:.2f}",
            )
        )
    return eval_results


def generate_model_card(model_name: str, scores_dict: Dict[str, List[float]]) -> ModelCard:
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
    card_data = ModelCardData(
        tags=["reinforcement-learning", *scores_dict.keys()],
        eval_results=generate_eval_results(scores_dict),
        model_name=model_name,
        datasets="gia-project/gia-dataset-parquet",
        pipeline_tag="reinforcement-learning",
    )
    card = ModelCard.from_template(
        card_data,
        template_path="templates/model_card.md",
        model_name=model_name,
        model_id="Gia2",
        tasks=[TASK_NAME_TO_ENV_ID[task_name] for task_name in scores_dict.keys()],
    )
    return card


def push_to_hub(model: PreTrainedModel, repo_id: str, scores_dict: Dict[str, List[float]], replay_path: str) -> None:
    """
    Push a model to the Hugging Face Hub.

    Args:
        path (`str`):
            Path to the model directory.
        repo_id (`str`):
            Repository ID to push to.
        scores_dict (`Dict[str, List[float]]`):
            Dictionary containing the scores for each task.
    """
    api = HfApi()

    # Create the repo
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Create a README.md using a template
    model_card = generate_model_card(repo_id, scores_dict)
    model_card.push_to_hub(repo_id, commit_message="Upload model card")

    # Push the model
    model.push_to_hub(repo_id, commit_message="Upload model")

    # As long as the the trainer does not use tokenizer, we mannually save it
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.push_to_hub(repo_id, commit_message="Upload tokenizer")

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
        while True:
            for i, it in enumerate(iterators):
                for _ in range(batch_size):
                    try:
                        yield next(it)
                    except StopIteration:
                        if stopping_strategy == "first_exhausted":
                            return
                        else:
                            iterators[i] = iter(datasets[i])
                            yield next(iterators[i])

    gen_kwargs = {"datasets": datasets, "batch_size": batch_size, "stopping_strategy": stopping_strategy}
    return IterableDataset.from_generator(generator=generator, gen_kwargs=gen_kwargs)
