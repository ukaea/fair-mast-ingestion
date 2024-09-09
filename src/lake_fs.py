import os
import argparse
import logging
from lakefs.client import Client
import lakefs
import subprocess
from datetime import datetime
import uuid
import sys

class LakeFSManager:
    def __init__(self, repo_name):
        self.client = Client()
        self.repo_name = repo_name
        self.repo = lakefs.Repository(repo_name, client=self.client)


    def execute_command(self, command):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info("Command output: %s", result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            logging.error("Error executing command: %s", e.stderr)
            return e

    def create_branch(self):
        logging.info("Creating branch for ingestion...")
        branch_name = f"branch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        try:
            branch = self.repo.branch(branch_name).create(source_reference="main")
            return branch
        except lakefs.exceptions.LakeFSException as e:
            logging.error(f"Failed to create branch: {e}")
            sys.exit(1)

    def upload_files_to_branch(self, branch_name, local_folder):
        logging.info("Uploading files from data directory to branch...")
        command = [
            "lakectl", "fs", "upload",
            f"lakefs://{self.repo_name}/{branch_name}/",
            "-s", local_folder,
            "-r"
        ]
        self.execute_command(command)

    def commit_branch(self, branch_name, message):
        logging.info("Committing changes to branch...")
        command = [
            "lakectl", "commit",
            f"lakefs://{self.repo_name}/{branch_name}/",
            "-m", message
        ]
        result = self.execute_command(command)

        if result.returncode != 0:
            logging.error(f"Commit failed for branch '{branch_name}'. Deleting branch and exiting.")
            self.delete_branch(branch_name)
            sys.exit(1)

    def show_diff(self, branch):
        main_branch = self.repo.branch("main")
        changes = list(main_branch.diff(other_ref=branch))
        logging.info(f"Number of changes made to main: {len(changes)}")

    def merge_branch(self, branch):
        logging.info("Merging branch to main...")
        main_branch = self.repo.branch("main")
        try:
            res = branch.merge_into(main_branch)
            return res
        except lakefs.exceptions.LakeFSException as e:
            logging.error(f"Failed to merge branch: {e}")
            sys.exit(1)

    def delete_branch(self, branch_name):
        logging.info("Deleting branch.")
        command = [
            "lakectl", "branch", "delete",
            f"lakefs://{self.repo_name}/{branch_name}",
            "--yes"
        ]
        self.execute_command(command)


def check_lakectl_installed():
    try:
        subprocess.run(["lakectl", "--version"], check=True, capture_output=True, text=True)
        logging.info("lakectl is installed.")
    except subprocess.CalledProcessError:
        logging.error("lakectl is not installed or not found in PATH.")
        sys.exit(1)


def validate_data_directory(local_folder):
    if not os.path.exists(local_folder):
        logging.error(f"Data directory '{local_folder}' does not exist.")
        sys.exit(1)
    elif not os.listdir(local_folder):
        logging.error(f"Data directory '{local_folder}' is empty.")
        sys.exit(1)
    logging.info(f"Data directory '{local_folder}' is valid.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="LakeFS Ingestion Script")
    parser.add_argument("--repo", required=True, help="Name of the LakeFS repository")
    parser.add_argument("--data-dir", required=True, help="Location of the data files to upload")
    parser.add_argument("--commit-message", default="Import data from CSD3", help="Commit message for the ingestion")
    args = parser.parse_args()

    check_lakectl_installed()
    validate_data_directory(args.data_dir)

    lakefs_manager = LakeFSManager(repo_name=args.repo)

    branch = lakefs_manager.create_branch()
    lakefs_manager.upload_files_to_branch(branch.id, args.data_dir)
    lakefs_manager.commit_branch(branch.id, args.commit_message)
    lakefs_manager.show_diff(branch)

    lakefs_manager.merge_branch(branch)
    lakefs_manager.delete_branch(branch.id)


if __name__ == "__main__":
    main()