import logging
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from vllm.distributed import cleanup_dist_env_and_memory


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }

    return args_dict


# taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/loggers/utils.py
def get_commit_from_path(repo_path: Union[Path, str]) -> Optional[str]:
    try:
        git_folder = Path(repo_path, ".git")
        if git_folder.is_file():
            git_folder = Path(
                git_folder.parent,
                git_folder.read_text(encoding="utf-8").split("\n")[0].split(" ")[-1],
            )
        if Path(git_folder, "HEAD").exists():
            head_name = (
                Path(git_folder, "HEAD")
                .read_text(encoding="utf-8")
                .split("\n")[0]
                .split(" ")[-1]
            )
            head_ref = Path(git_folder, head_name)
            git_hash = head_ref.read_text(encoding="utf-8").replace("\n", "")
        else:
            git_hash = None
    except Exception as err:
        logging.debug(
            f"Failed to retrieve a Git commit hash from path: {str(repo_path)}. Error: {err}"
        )
        return None
    return git_hash


def get_git_commit_hash():
    """
    Gets the git commit hash of your current repo (if it exists).
    Source: https://github.com/EleutherAI/gpt-neox/blob/b608043be541602170bfcfb8ec9bf85e8a0799e0/megatron/neox_arguments/neox_args.py#L42
    """
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
        git_hash = git_hash.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # FileNotFoundError occurs when git not installed on system
        git_hash = get_commit_from_path(os.getcwd())  # git hash of repo if exists
    return git_hash


def get_error_count(
    errors: pd.Series,
):
    error_df = pd.DataFrame(errors.to_list())

    counter = Counter({})
    for _, value in error_df.items():
        counts = Counter(value.value_counts().to_dict())
        counter.update(counts)

    return dict(counter)


def delete_vllm_model_and_free_memory(llm):
    try:
        # v1 vllm
        del llm.llm.llm_engine.engine_core
    except:
        del llm.llm.llm_engine.model_executor

    del llm.llm
    del llm

    cleanup_dist_env_and_memory(shutdown_ray=True)
