#!/usr/bin/env python3
"""Build reproducible container images used by Triton CPU CI on s390x."""

import argparse
import functools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DOCKERFILE = SCRIPT_DIR / "Dockerfile"

DEFAULT_IMAGE_REPOSITORY = "ghcr.io/triton-lang/triton-cpu-s390x-ci"
IMAGE_REPOSITORY_ENV = "TRITON_CPU_S390X_CI_IMAGE_REPOSITORY"
DEFAULT_ENGINE = "podman"

REQUIRED_BUILD_ARGUMENTS = (
    "BASE_IMAGE",
    "PYTORCH_VERSION",
    "PYTORCH_INDEX_URL",
)

UBUNTU_26_04 = "docker.io/library/ubuntu@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c"

CONFIGURATIONS = {
    # The OS of the actionspz IBM Z runner image, with PyTorch's CPU wheel for
    # linux_s390x (cp314).
    "ubuntu26.04-pytorch-2.15": {
        "tag": "ubuntu26.04-llvm{llvm_short}-pytorch2.15.0-r1",
        "build_args": {
            "BASE_IMAGE": UBUNTU_26_04,
            "PYTORCH_VERSION": "2.15.0+cpu",
            "PYTORCH_INDEX_URL": "https://download.pytorch.org/whl/cpu",
        },
    },
}


# Derived from the repository so the image cannot drift from Triton's pins.
@functools.cache
def pinned_build_arguments() -> dict[str, str]:
    llvm_info = json.loads((REPOSITORY_ROOT / "cmake" / "llvm-info.json").read_text())
    json_version = (REPOSITORY_ROOT / "cmake" / "json-version.txt").read_text().strip()
    cpu_cmake = (REPOSITORY_ROOT / "third_party" / "cpu" / "CMakeLists.txt").read_text()
    sleef = re.search(r"GIT_REPOSITORY https://github\.com/shibatch/sleef\.git\s+GIT_TAG ([0-9a-f]{40})", cpu_cmake)
    if sleef is None:
        raise ValueError("cannot find the SLEEF GIT_TAG in third_party/cpu/CMakeLists.txt")
    return {
        "LLVM_COMMIT_HASH": llvm_info["llvm_hash"],
        "JSON_VERSION": json_version,
        "SLEEF_COMMIT": sleef.group(1),
    }


def validate_configurations() -> None:
    required_arguments = set(REQUIRED_BUILD_ARGUMENTS)
    tags = set()

    for name, configuration in CONFIGURATIONS.items():
        if not isinstance(name, str) or not name:
            raise ValueError("configuration names must be nonempty strings")
        if not isinstance(configuration, dict):
            raise ValueError(f"{name}: configuration must be a dictionary")

        tag = configuration.get("tag")
        if not isinstance(tag, str) or not tag:
            raise ValueError(f"{name}: tag must be a nonempty string")
        if tag in tags:
            raise ValueError(f"{name}: duplicate tag: {tag}")
        tags.add(tag)

        build_args = configuration.get("build_args")
        if not isinstance(build_args, dict):
            raise ValueError(f"{name}: build_args must be a dictionary")

        actual_arguments = set(build_args)
        missing = required_arguments - actual_arguments
        unexpected = actual_arguments - required_arguments
        if missing:
            raise ValueError(f"{name}: missing build arguments: {', '.join(sorted(missing))}")
        if unexpected:
            raise ValueError(f"{name}: unexpected build arguments: "
                             f"{', '.join(sorted(unexpected))}")
        for argument, value in build_args.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}: {argument} must be a nonempty string")
        if "@sha256:" not in build_args["BASE_IMAGE"]:
            raise ValueError(f"{name}: BASE_IMAGE must be pinned by digest")


def resolved_build_arguments(configuration_name: str) -> dict[str, str]:
    return {**CONFIGURATIONS[configuration_name]["build_args"], **pinned_build_arguments()}


def image_tag(configuration_name: str) -> str:
    llvm_short = pinned_build_arguments()["LLVM_COMMIT_HASH"][:8]
    return CONFIGURATIONS[configuration_name]["tag"].format(llvm_short=llvm_short)


def image_reference(configuration_name: str, image_repository: str) -> str:
    return f"{image_repository}:{image_tag(configuration_name)}"


def build_command(configuration_name: str, image_repository: str, engine: str) -> list[str]:
    # The llvm and wheels stages are independent; let podman run them together.
    command = [engine, "build", "--jobs", "2"] if engine == "podman" else [engine, "build"]
    command.extend(["--file", str(DOCKERFILE)])
    for argument, value in resolved_build_arguments(configuration_name).items():
        command.extend(["--build-arg", f"{argument}={value}"])
    command.extend([
        "--tag",
        image_reference(configuration_name, image_repository),
        str(REPOSITORY_ROOT),
    ])
    return command


def print_command(command: list[str]) -> None:
    print(shlex.join(command))


def list_configurations() -> None:
    pins = pinned_build_arguments()
    for name, configuration in CONFIGURATIONS.items():
        build_args = configuration["build_args"]
        print(name)
        print(f"  tag: {image_tag(name)}")
        print(f"  LLVM: {pins['LLVM_COMMIT_HASH']}")
        print(f"  PyTorch: {build_args['PYTORCH_VERSION']}")


def show_configuration(name: str, image_repository: str) -> None:
    print(f"name: {name}")
    print(f"image: {image_reference(name, image_repository)}")
    for argument, value in resolved_build_arguments(name).items():
        print(f"{argument}={value}")


def build_configuration(name: str, image_repository: str, engine: str, dry_run: bool) -> None:
    command = build_command(name, image_repository, engine)
    print_command(command)
    if not dry_run:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        image = image_reference(name, image_repository)
        print(f"Built {image}")
        print(f"Publish manually with: {engine} push {image}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build container images used by Triton CPU CI on s390x.")
    parser.add_argument(
        "--image-repository",
        default=os.environ.get(IMAGE_REPOSITORY_ENV, DEFAULT_IMAGE_REPOSITORY),
        help=("image repository "
              f"(default: ${IMAGE_REPOSITORY_ENV} or "
              f"{DEFAULT_IMAGE_REPOSITORY})"),
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=("podman", "docker"),
        help=f"container engine (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print build commands without running them",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list available configurations")

    show_parser = subparsers.add_parser("show", help="show one configuration")
    show_parser.add_argument("configuration", choices=CONFIGURATIONS)

    build_parser = subparsers.add_parser("build", help="build one configuration")
    build_parser.add_argument("configuration", choices=CONFIGURATIONS)

    subparsers.add_parser("build-all", help="build every configuration")
    return parser.parse_args()


def main() -> int:
    try:
        validate_configurations()
        args = parse_arguments()

        if args.command == "list":
            list_configurations()
        elif args.command == "show":
            show_configuration(args.configuration, args.image_repository)
        elif args.command == "build":
            build_configuration(args.configuration, args.image_repository, args.engine, args.dry_run)
        elif args.command == "build-all":
            for name in CONFIGURATIONS:
                build_configuration(name, args.image_repository, args.engine, args.dry_run)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
