import json
import os
import random
import sys
import tempfile
from contextlib import contextmanager
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from arch.bootstrap import IIDBootstrap
from datasets import IterableDataset
from huggingface_hub import EvalResult, HfApi, ModelCard, ModelCardData
from scipy.stats import trim_mean
from transformers import PreTrainedModel, ProcessorMixin


PRETTY_TASK_NAMES = {
    "atari-alien": "Alien-v5",
    "atari-amidar": "Amidar-v5",
    "atari-assault": "Assault-v5",
    "atari-asterix": "Asterix-v5",
    "atari-asteroids": "Asteroids-v5",
    "atari-atlantis": "Atlantis-v5",
    "atari-bankheist": "BankHeist-v5",
    "atari-battlezone": "BattleZone-v5",
    "atari-beamrider": "BeamRider-v5",
    "atari-berzerk": "Berzerk-v5",
    "atari-bowling": "Bowling-v5",
    "atari-boxing": "Boxing-v5",
    "atari-breakout": "Breakout-v5",
    "atari-centipede": "Centipede-v5",
    "atari-choppercommand": "ChopperCommand-v5",
    "atari-crazyclimber": "CrazyClimber-v5",
    "atari-defender": "Defender-v5",
    "atari-demonattack": "DemonAttack-v5",
    "atari-doubledunk": "DoubleDunk-v5",
    "atari-enduro": "Enduro-v5",
    "atari-fishingderby": "FishingDerby-v5",
    "atari-freeway": "Freeway-v5",
    "atari-frostbite": "Frostbite-v5",
    "atari-gopher": "Gopher-v5",
    "atari-gravitar": "Gravitar-v5",
    "atari-hero": "Hero-v5",
    "atari-icehockey": "IceHockey-v5",
    "atari-jamesbond": "Jamesbond-v5",
    "atari-kangaroo": "Kangaroo-v5",
    "atari-krull": "Krull-v5",
    "atari-kungfumaster": "KungFuMaster-v5",
    "atari-montezumarevenge": "MontezumaRevenge-v5",
    "atari-mspacman": "MsPacman-v5",
    "atari-namethisgame": "NameThisGame-v5",
    "atari-phoenix": "Phoenix-v5",
    "atari-pitfall": "Pitfall-v5",
    "atari-pong": "Pong-v5",
    "atari-privateeye": "PrivateEye-v5",
    "atari-qbert": "Qbert-v5",
    "atari-riverraid": "Riverraid-v5",
    "atari-roadrunner": "RoadRunner-v5",
    "atari-robotank": "Robotank-v5",
    "atari-seaquest": "Seaquest-v5",
    "atari-skiing": "Skiing-v5",
    "atari-solaris": "Solaris-v5",
    "atari-spaceinvaders": "SpaceInvaders-v5",
    "atari-stargunner": "StarGunner-v5",
    "atari-surround": "Surround-v5",
    "atari-tennis": "Tennis-v5",
    "atari-timepilot": "TimePilot-v5",
    "atari-tutankham": "Tutankham-v5",
    "atari-upndown": "UpNDown-v5",
    "atari-venture": "Venture-v5",
    "atari-videopinball": "VideoPinball-v5",
    "atari-wizardofwor": "WizardOfWor-v5",
    "atari-yarsrevenge": "YarsRevenge-v5",
    "atari-zaxxon": "Zaxxon-v5",
    "babyai-action-obj-door": "ActionObjDoor-v0",
    "babyai-blocked-unlock-pickup": "BlockedUnlockPickup-v0",
    "babyai-boss-level-no-unlock": "BossLevelNoUnlock-v0",
    "babyai-boss-level": "BossLevel-v0",
    "babyai-find-obj-s5": "FindObjS5-v0",
    "babyai-go-to-door": "GoToDoor-v0",
    "babyai-go-to-imp-unlock": "GoToImpUnlock-v0",
    "babyai-go-to-local": "GoToLocal-v0",
    "babyai-go-to-obj-door": "GoToObjDoor-v0",
    "babyai-go-to-obj": "GoToObj-v0",
    "babyai-go-to-red-ball-grey": "GoToRedBallGrey-v0",
    "babyai-go-to-red-ball-no-dists": "GoToRedBallNoDists-v0",
    "babyai-go-to-red-ball": "GoToRedBall-v0",
    "babyai-go-to-red-blue-ball": "GoToRedBlueBall-v0",
    "babyai-go-to-seq": "GoToSeq-v0",
    "babyai-go-to": "GoTo-v0",
    "babyai-key-corridor": "KeyCorridor-v0",
    "babyai-mini-boss-level": "MiniBossLevel-v0",
    "babyai-move-two-across-s8n9": "MoveTwoAcrossS8N9-v0",
    "babyai-one-room-s8": "OneRoomS8-v0",
    "babyai-open-door": "OpenDoor-v0",
    "babyai-open-doors-order-n4": "OpenDoorsOrderN4-v0",
    "babyai-open-red-door": "OpenRedDoor-v0",
    "babyai-open-two-doors": "OpenTwoDoors-v0",
    "babyai-open": "Open-v0",
    "babyai-pickup-above": "PickupAbove-v0",
    "babyai-pickup-dist": "PickupDist-v0",
    "babyai-pickup-loc": "PickupLoc-v0",
    "babyai-pickup": "Pickup-v0",
    "babyai-put-next-local": "PutNextLocal-v0",
    "babyai-put-next": "PutNextS7N4-v0",
    "babyai-synth-loc": "SynthLoc-v0",
    "babyai-synth-seq": "SynthSeq-v0",
    "babyai-synth": "Synth-v0",
    "babyai-unblock-pickup": "UnblockPickup-v0",
    "babyai-unlock-local": "UnlockLocal-v0",
    "babyai-unlock-pickup": "UnlockPickup-v0",
    "babyai-unlock-to-unlock": "UnlockToUnlock-v0",
    "babyai-unlock": "Unlock-v0",
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

PRETTY_DOMAIN_NAMES = {"atari": "Atari 57", "babyai": "BabyAI", "metaworld": "MetaWorld", "mujoco": "MuJoCo"}


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


def normalize(values: np.ndarray, env_id: Optional[str], strategy: str) -> np.ndarray:
    """
    Normalize the scores.

    Args:
        values (np.ndarray): Scores to normalize.
        env_id (str, optional): Environment name.
        strategy (str): Normalization strategy. Can be either "max" or "expert" or "human".

    Returns:
        np.ndarray: Normalized scores.
    """
    with open("gia/eval/rl/scores_dict.json", "r") as f:
        scores_dict = json.load(f)

    # Check if the environment is available
    if env_id not in scores_dict:
        raise KeyError(f"Environment {env_id} not found in scores_dict.json")
    if "random" not in scores_dict[env_id]:
        raise KeyError(f"Random scores not found for environment {env_id}")
    random_score = scores_dict[env_id]["random"]["mean"]

    # Get the max score depending on the strategy
    if strategy == "max":
        max_score = np.max(values)
    elif strategy == "expert":
        if "expert" not in scores_dict[env_id]:
            raise KeyError(f"Expert scores not found for environment {env_id}")
        max_score = scores_dict[env_id]["expert"]["mean"]
    elif strategy == "human":
        if "human" not in scores_dict[env_id]:
            raise KeyError(f"Human scores not found for environment {env_id}")
        max_score = scores_dict[env_id]["human"]["mean"]

    return [(v - random_score) / (max_score - random_score) for v in values]


def iqm(x):
    return trim_mean(x, proportiontocut=0.25)


def stratified_with_ci(data_list, func):
    """
    Calculate the stratified interquartile mean and confidence interval of a list of datasets.

    Args:
        data_list (list of list of float): List of datasets. Each dataset is a list of scores.

    Returns:
        np.ndarray: Confidence interval of shape (2,).
    """
    # Convert the list of lists into a DataFrame
    data = []
    for i, dataset in enumerate(data_list):
        for v in dataset:
            data.append({"dataset": i, "val": v})
    data = pd.DataFrame(data)

    # Bootstrap
    bs = IIDBootstrap(data)
    def stratified_func(d):
        return d.groupby("dataset")["val"].apply(func).mean()
    ci = bs.conf_int(stratified_func, 1000, method="percentile")
    val = stratified_func(data)
    return val, ci[:, 0]


def stratified_iqm_with_ci(data_list: List[List[float]]) -> np.ndarray:
    """
    Calculate the stratified interquartile mean and confidence interval of a list of datasets.

    Args:
        data_list (list of list of float): List of datasets. Each dataset is a list of scores.

    Returns:
        np.ndarray: Confidence interval of shape (2,).
    """
    return stratified_with_ci(data_list, iqm)


def generate_rl_eval_results(evaluations: Dict[str, List[float]]) -> List[EvalResult]:
    """
    Generate a list of EvalResult objects.

    Args:
        evaluations (`Dict[str, List[float]]`):
            Dictionary containing the evaluation results for each task.

    Returns:
        `List[EvalResult]`:
            A list of EvalResult objects.
    """
    eval_results = []

    # Aggregate the results
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        domain_scores = {
            task_name: scores for task_name, scores in evaluations.items() if task_name.startswith(domain)
        }
        # Normalize the scores
        norm_scores = {
            task_name: normalize(np.array(scores), task_name, "expert") for task_name, scores in domain_scores.items()
        }

        # Compute the stratified interquartile mean and confidence interval
        mean_scores, ci = stratified_iqm_with_ci(list(norm_scores.values()))

        eval_results.append(
            EvalResult(
                task_type="reinforcement-learning",
                task_name="Reinforcement Learning",
                dataset_type=domain,
                dataset_name=PRETTY_DOMAIN_NAMES[domain],
                metric_type="iqm_expert_normalized_total_reward",
                metric_name="IQM expert normalized total reward",
                metric_value=f"{mean_scores:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
            )
        )

    atari_scores = {task_name: scores for task_name, scores in evaluations.items() if task_name.startswith("atari")}
    # Normalize the scores
    norm_scores = {
        task_name: normalize(np.array(scores), task_name, "human") for task_name, scores in atari_scores.items()
    }
    # Compute the stratified interquartile mean and confidence interval
    mean_scores, ci = stratified_iqm_with_ci(list(norm_scores.values()))

    eval_results.append(
        EvalResult(
            task_type="reinforcement-learning",
            task_name="Reinforcement Learning",
            dataset_type="atari",
            dataset_name=PRETTY_DOMAIN_NAMES["atari"],
            metric_type="iqm_human_normalized_total_reward",
            metric_name="IQM human normalized total reward",
            metric_value=f"{mean_scores:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
        )
    )

    for task_name, scores in evaluations.items():
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

    for task_name, scores in evaluations.items():
        norm_scores = normalize(np.array(scores), task_name, "expert")
        mean_scores = np.mean(norm_scores)
        std_scores = np.std(norm_scores)

        eval_results.append(
            EvalResult(
                task_type="reinforcement-learning",
                task_name="Reinforcement Learning",
                dataset_type=task_name,
                dataset_name=PRETTY_TASK_NAMES[task_name],
                metric_type="expert_normalized_total_reward",
                metric_name="Expert normalized total reward",
                metric_value=f"{mean_scores:.2f} +/- {std_scores:.2f}",
            )
        )

    for task_name, scores in evaluations.items():
        if not task_name.startswith("atari"):
            continue
        norm_scores = normalize(np.array(scores), task_name, "human")
        mean_scores = np.mean(norm_scores)
        std_scores = np.std(norm_scores)

        eval_results.append(
            EvalResult(
                task_type="reinforcement-learning",
                task_name="Reinforcement Learning",
                dataset_type=task_name,
                dataset_name=PRETTY_TASK_NAMES[task_name],
                metric_type="human_normalized_total_reward",
                metric_name="Human normalized total reward",
                metric_value=f"{mean_scores:.2f} +/- {std_scores:.2f}",
            )
        )

    return eval_results


def generate_model_card(model_name: str, evaluations: Optional[Dict[str, List[float]]] = None) -> ModelCard:
    """
    Generate a ModelCard from a template.

    Args:
        model_name (`str`):
            Model name.
        evaluations (`Dict[str, List[float]]`):
            Dictionary containing the evaluation results for each task.

    Returns:
        `ModelCard`:
            A ModelCard object.
    """
    tags = ["reinforcement-learning"]
    if evaluations is not None:
        tags.extend(evaluations.keys())
    card_data = ModelCardData(
        tags=tags,
        eval_results=generate_rl_eval_results(evaluations) if evaluations is not None else None,
        model_name=model_name,
        datasets="gia-project/gia-dataset",
        pipeline_tag="reinforcement-learning",
    )
    card = ModelCard.from_template(
        card_data,
        template_path="templates/model_card.md",
        model_name=model_name,
        model_id="Gia",
        tasks=[PRETTY_TASK_NAMES[task_name] for task_name in evaluations.keys()] if evaluations is not None else [],
    )
    return card


def push_to_hub(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    repo_id: str,
    replay_path: Optional[str] = None,
    eval_path: Optional[str] = None,
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
        replay_path (`str` or `None`, **optional**):
            Path to the replay video.
        eval_path (`str` or `None`, **optional**):
            Path to the evaluation scores.
    """
    api = HfApi()

    # Create the repo
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Get the evaluation scores to compute the mean and std
    if eval_path is not None:
        with open(eval_path, "r") as file:
            evaluations = json.load(file)
    else:
        evaluations = None

    # Create a README.md using a template
    model_card = generate_model_card(repo_id, evaluations)
    model_card.push_to_hub(repo_id, commit_message="Upload model card")

    # Push the model
    model.push_to_hub(repo_id, commit_message="Upload model")

    # Push the processor
    processor.push_to_hub(repo_id, commit_message="Upload processor")

    # Push the replay
    if replay_path is not None:
        api.upload_file(
            path_or_fileobj=replay_path,
            path_in_repo="replay.mp4",
            repo_id=repo_id,
            commit_message="Upload replay",
            repo_type="model",
        )

    # Push the evaluation scores
    if eval_path is not None:
        api.upload_file(
            path_or_fileobj=eval_path,
            path_in_repo="evaluations.json",
            repo_id=repo_id,
            commit_message="Upload evaluations",
            repo_type="model",
        )

    print(f"Pushed model to  \033[34mhttps://huggingface.co/{repo_id}\033[0m")


def save_video_grid(
    videos: List[List[np.ndarray]],
    input_fps: List[int],
    output_filename: str = "output.mp4",
    width: int = 1920,
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
    datasets: List[IterableDataset],
    batch_size: int,
    stopping_strategy: str = "all_exhausted",
    weights: List[float] = None,
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
        weights (`List[float]`, **optional**):
            List of weights for each dataset. If None, uniform weights are used.

    Returns:
        `IterableDataset`:
            A mixed IterableDataset object.
    """

    def generator(
        datasets: List[IterableDataset],
        batch_size: int,
        stopping_strategy: str = "all_exhausted",
        weights: List[float] = None,
    ):
        assert stopping_strategy in ["first_exhausted", "all_exhausted"]
        iterators = [iter(dataset) for dataset in datasets]
        exhausted = [False] * len(datasets)  # A list to keep track of which iterators are exhausted
        weights = weights if weights is not None else [1.0] * len(datasets)
        should_stop = False
        while not should_stop:
            dataset_idx = random.choices(range(len(datasets)), weights=weights, k=1)[0]  # Choose a dataset randomly
            iterator = iterators[dataset_idx]
            for _ in range(batch_size):
                try:
                    yield next(iterator)
                except StopIteration:
                    if stopping_strategy == "first_exhausted":
                        should_stop = True
                    else:
                        # Mark the iterator as exhausted
                        exhausted[dataset_idx] = True
                        # Check if all iterators are exhausted
                        if all(exhausted):
                            should_stop = True
                    # Reinitialize the exhausted iterator
                    iterator = iterators[dataset_idx] = iter(datasets[dataset_idx])
                    yield next(iterators[dataset_idx])

    gen_kwargs = {
        "datasets": datasets,
        "batch_size": batch_size,
        "stopping_strategy": stopping_strategy,
        "weights": weights,
    }
    return IterableDataset.from_generator(generator=generator, gen_kwargs=gen_kwargs)
