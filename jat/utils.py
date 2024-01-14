import json
import os
import tempfile
from typing import Dict, List, Optional

import cv2
import numpy as np
from huggingface_hub import EvalResult, HfApi, ModelCard, ModelCardData
from rliable import library as rly
from rliable import metrics
from transformers import PreTrainedModel, ProcessorMixin


PRETTY_TASK_NAMES = {
    "atari-alien": "Alien",
    "atari-amidar": "Amidar",
    "atari-assault": "Assault",
    "atari-asterix": "Asterix",
    "atari-asteroids": "Asteroids",
    "atari-atlantis": "Atlantis",
    "atari-bankheist": "Bank Heist",
    "atari-battlezone": "Battle Zone",
    "atari-beamrider": "Beam Rider",
    "atari-berzerk": "Berzerk",
    "atari-bowling": "Bowling",
    "atari-boxing": "Boxing",
    "atari-breakout": "Breakout",
    "atari-centipede": "Centipede",
    "atari-choppercommand": "Chopper Command",
    "atari-crazyclimber": "Crazy Climber",
    "atari-defender": "Defender",
    "atari-demonattack": "Demon Attack",
    "atari-doubledunk": "Double Dunk",
    "atari-enduro": "Enduro",
    "atari-fishingderby": "Fishing Derby",
    "atari-freeway": "Freeway",
    "atari-frostbite": "Frostbite",
    "atari-gopher": "Gopher",
    "atari-gravitar": "Gravitar",
    "atari-hero": "H.E.R.O.",
    "atari-icehockey": "Ice Hockey",
    "atari-jamesbond": "James Bond",
    "atari-kangaroo": "Kangaroo",
    "atari-krull": "Krull",
    "atari-kungfumaster": "Kung-Fu Master",
    "atari-montezumarevenge": "Montezuma's Revenge",
    "atari-mspacman": "Ms. Pacman",
    "atari-namethisgame": "Name This Game",
    "atari-phoenix": "Phoenix",
    "atari-pitfall": "PitFall",
    "atari-pong": "Pong",
    "atari-privateeye": "Private Eye",
    "atari-qbert": "Q*Bert",
    "atari-riverraid": "River Raid",
    "atari-roadrunner": "Road Runner",
    "atari-robotank": "Robotank",
    "atari-seaquest": "Seaquest",
    "atari-skiing": "Skiing",
    "atari-solaris": "Solaris",
    "atari-spaceinvaders": "Space Invaders",
    "atari-stargunner": "Star Gunner",
    "atari-surround": "Surround",
    "atari-tennis": "Tennis",
    "atari-timepilot": "Time Pilot",
    "atari-tutankham": "Tutankham",
    "atari-upndown": "Up and Down",
    "atari-venture": "Venture",
    "atari-videopinball": "Video Pinball",
    "atari-wizardofwor": "Wizard of Wor",
    "atari-yarsrevenge": "Yars Revenge",
    "atari-zaxxon": "Zaxxon",
    "babyai-action-obj-door": "Action Obj Door",
    "babyai-blocked-unlock-pickup": "Blocked Unlock Pickup",
    "babyai-boss-level-no-unlock": "Boss Level No Unlock",
    "babyai-boss-level": "Boss Level",
    "babyai-find-obj-s5": "Find Obj S5",
    "babyai-go-to-door": "Go To Door",
    "babyai-go-to-imp-unlock": "Go To Imp Unlock",
    "babyai-go-to-local": "Go To Local",
    "babyai-go-to-obj-door": "Go To Obj Door",
    "babyai-go-to-obj": "Go To Obj",
    "babyai-go-to-red-ball-grey": "Go To Red Ball Grey",
    "babyai-go-to-red-ball-no-dists": "Go To Red Ball No Dists",
    "babyai-go-to-red-ball": "Go To Red Ball",
    "babyai-go-to-red-blue-ball": "Go To Red Blue Ball",
    "babyai-go-to-seq": "Go To Seq",
    "babyai-go-to": "Go To",
    "babyai-key-corridor": "Key Corridor",
    "babyai-mini-boss-level": "Mini Boss Level",
    "babyai-move-two-across-s8n9": "Move Two Across S8N9",
    "babyai-one-room-s8": "One Room S8",
    "babyai-open-door": "Open Door",
    "babyai-open-doors-order-n4": "Open Doors Order N4",
    "babyai-open-red-door": "Open Red Door",
    "babyai-open-two-doors": "Open Two Doors",
    "babyai-open": "Open",
    "babyai-pickup-above": "Pickup Above",
    "babyai-pickup-dist": "Pickup Dist",
    "babyai-pickup-loc": "Pickup Loc",
    "babyai-pickup": "Pickup",
    "babyai-put-next-local": "Put Next Local",
    "babyai-put-next": "Put Next S7N4",
    "babyai-synth-loc": "Synth Loc",
    "babyai-synth-seq": "Synth Seq",
    "babyai-synth": "Synth",
    "babyai-unblock-pickup": "Unblock Pickup",
    "babyai-unlock-local": "Unlock Local",
    "babyai-unlock-pickup": "Unlock Pickup",
    "babyai-unlock-to-unlock": "Unlock To Unlock",
    "babyai-unlock": "Unlock",
    "conceptual-captions": "Conceptual Captions",
    "metaworld-assembly": "Assembly",
    "metaworld-basketball": "Basketball",
    "metaworld-bin-picking": "BinPicking",
    "metaworld-box-close": "Box Close",
    "metaworld-button-press-topdown-wall": "Button Press Topdown Wall",
    "metaworld-button-press-topdown": "Button Press Topdown",
    "metaworld-button-press-wall": "Button Press Wall",
    "metaworld-button-press": "Button Press",
    "metaworld-coffee-button": "Coffee Button",
    "metaworld-coffee-pull": "Coffee Pull",
    "metaworld-coffee-push": "Coffee Push",
    "metaworld-dial-turn": "Dial Turn",
    "metaworld-disassemble": "Disassemble",
    "metaworld-door-close": "Door Close",
    "metaworld-door-lock": "Door Lock",
    "metaworld-door-open": "Door Open",
    "metaworld-door-unlock": "Door Unlock",
    "metaworld-drawer-close": "Drawer Close",
    "metaworld-drawer-open": "Drawer Open",
    "metaworld-faucet-close": "Faucet Close",
    "metaworld-faucet-open": "Faucet Open",
    "metaworld-hammer": "Hammer",
    "metaworld-hand-insert": "Hand Insert",
    "metaworld-handle-press-side": "Handle Press Side",
    "metaworld-handle-press": "Handle Press",
    "metaworld-handle-pull-side": "Handle Pull Side",
    "metaworld-handle-pull": "Handle Pull",
    "metaworld-lever-pull": "Lever Pull",
    "metaworld-peg-insert-side": "Peg Insert Side",
    "metaworld-peg-unplug-side": "Peg Unplug Side",
    "metaworld-pick-out-of-hole": "Pick Out Of Hole",
    "metaworld-pick-place-wall": "Pick Place Wall",
    "metaworld-pick-place": "Pick Place",
    "metaworld-plate-slide-back-side": "Plate Slide Back Side",
    "metaworld-plate-slide-back": "Plate Slide Back",
    "metaworld-plate-slide-side": "Plate Slide Side",
    "metaworld-plate-slide": "Plate Slide",
    "metaworld-push-back": "Push Back",
    "metaworld-push-wall": "Push Wall",
    "metaworld-push": "Push",
    "metaworld-reach-wall": "Reach Wall",
    "metaworld-reach": "Reach",
    "metaworld-shelf-place": "Shelf Place",
    "metaworld-soccer": "Soccer",
    "metaworld-stick-pull": "Stick Pull",
    "metaworld-stick-push": "Stick Push",
    "metaworld-sweep-into": "Sweep Into",
    "metaworld-sweep": "Sweep",
    "metaworld-window-close": "Window Close",
    "metaworld-window-open": "Window Open",
    "mujoco-ant": "Ant",
    "mujoco-doublependulum": "Inverted Double Pendulum",
    "mujoco-halfcheetah": "Half Cheetah",
    "mujoco-hopper": "Hopper",
    "mujoco-humanoid": "Humanoid",
    "mujoco-pendulum": "Inverted Pendulum",
    "mujoco-pusher": "Pusher",
    "mujoco-reacher": "Reacher",
    "mujoco-standup": "Humanoid Standup",
    "mujoco-swimmer": "Swimmer",
    "mujoco-walker": "Walker 2d",
    "ok-vqa": "OK-VQA",
    "oscar": "OSCAR",
    "wikipedia": "Wikipedia",
}

PRETTY_DOMAIN_NAMES = {"atari": "Atari 57", "babyai": "BabyAI", "metaworld": "MetaWorld", "mujoco": "MuJoCo"}


def get_scores_dict() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Get the scores dictionary.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Dictionary containing the scores for each task.
    """
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval", "rl", "scores_dict.json"))

    # Now you can use this path to open and read the JSON file
    with open(file_path, "r") as file:
        scores_dict = json.load(file)

    return scores_dict


def normalize(values: List[float], env_id: str, strategy: str) -> List[float]:
    """
    Normalize the scores.

    Args:
        values (List[float]): Scores to normalize.
        env_id (str): Environment name.
        strategy (str): Normalization strategy. Can be either "max" or "expert" or "human".

    Returns:
        List[float]: Normalized scores.
    """
    scores_dict = get_scores_dict()

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

    if max_score <= random_score:
        print(env_id)
        return [1.0 for v in values]
    # if "human" in scores_dict[env_id] and scores_dict[env_id]["human"]["mean"] > scores_dict[env_id]["expert"]["mean"]:
    #     print(env_id)
    #     return None
    return [(v - random_score) / abs(max_score - random_score) for v in values]


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
        # Exlcude None
        norm_scores = {k: v for k, v in norm_scores.items() if v is not None}

        # Compute the stratified interquartile mean and confidence interval
        scores_dict = {"a": np.array(list(norm_scores.values())).T}

        def aggregate_func(x):
            return np.array([metrics.aggregate_iqm(x)])

        aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(scores_dict, aggregate_func)
        iqm, low, high = aggregate_scores["a"][0], aggregate_score_cis["a"][0][0], aggregate_score_cis["a"][1][0]

        eval_results.append(
            EvalResult(
                task_type="reinforcement-learning",
                task_name="Reinforcement Learning",
                dataset_type=domain,
                dataset_name=PRETTY_DOMAIN_NAMES[domain],
                metric_type="iqm_expert_normalized_total_reward",
                metric_name="IQM expert normalized total reward",
                metric_value=f"{iqm:.2f} [{low:.2f}, {high:.2f}]",
            )
        )

    atari_scores = {task_name: scores for task_name, scores in evaluations.items() if task_name.startswith("atari")}

    # Normalize the scores
    norm_scores = {
        task_name: normalize(np.array(scores), task_name, "human") for task_name, scores in atari_scores.items()
    }

    # Compute the stratified interquartile mean and confidence interval
    scores_dict = {"a": np.array(list(norm_scores.values())).T}

    def aggregate_func(x):
        return np.array([metrics.aggregate_iqm(x)])

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(scores_dict, aggregate_func)
    iqm, low, high = aggregate_scores["a"][0], aggregate_score_cis["a"][0][0], aggregate_score_cis["a"][1][0]

    eval_results.append(
        EvalResult(
            task_type="reinforcement-learning",
            task_name="Reinforcement Learning",
            dataset_type="atari",
            dataset_name=PRETTY_DOMAIN_NAMES["atari"],
            metric_type="iqm_human_normalized_total_reward",
            metric_name="IQM human normalized total reward",
            metric_value=f"{iqm:.2f} [{low:.2f}, {high:.2f}]",
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
        if norm_scores is None:
            continue
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
        datasets="jat-project/jat-dataset",
        pipeline_tag="reinforcement-learning",
    )
    card = ModelCard.from_template(
        card_data,
        template_path="templates/model_card.md",
        model_name=model_name,
        model_id="Jat",
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
