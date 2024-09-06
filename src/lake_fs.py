import os
import argparse
from lakefs.client import Client
import lakefs
import subprocess
from datetime import datetime

class LakeFSManager:
    def __init__(self, repo_name):
        self.client = Client()
        self.repo_name = repo_name
        self.repo = lakefs.Repository(repo_name, client=self.client)

    def execute_command(self, command):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Output:", result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print("An error occurred while executing the command.")
            print("Error message:", e.stderr)
            return e

    def create_branch(self):
        print("Creating branch for ingestion...")
        branch_name = datetime.now().strftime("branch-%Y%m%d-%H%M%S")
        branch = self.repo.branch(branch_name).create(source_reference="main")
        return branch

    def upload_files_to_branch(self, branch_name, local_folder):
        print("Uploading files from data directory to branch...")
        command = [
            "lakectl", "fs", "upload",
            f"lakefs://{self.repo_name}/{branch_name}/",
            "-s", local_folder,
            "-r"
        ]
        self.execute_command(command)

    def commit_branch(self, branch_name, message):
        print("Committing changes to branch...")
        command = [
            "lakectl", "commit",
            f"lakefs://{self.repo_name}/{branch_name}/",
            "-m", message
        ]
        result = self.execute_command(command)

        if result.returncode != 0:
            print(f"Commit failed for branch '{branch_name}'. Deleting branch and exiting.")
            self.delete_branch(branch_name)
            exit()

    def show_diff(self, branch):
        main_branch = self.repo.branch("main")
        changes = list(main_branch.diff(other_ref=branch))
        print(f"Number of changes made to main: {len(changes)}")

    def merge_branch(self, branch):
        print("Merging branch to main...")
        main_branch = self.repo.branch("main")
        res = branch.merge_into(main_branch)
        return res

    def delete_branch(self, branch_name):
        print("Deleting branch.")
        command = [
            "lakectl", "branch", "delete",
            f"lakefs://{self.repo_name}/{branch_name}",
            "--yes"
        ]
        self.execute_command(command)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="LakeFS Ingestion Script")
    parser.add_argument("--repo", required=True, help="Name of the LakeFS repository")
    parser.add_argument("--data-dir", required=True, help="Location of the data files to upload")
    parser.add_argument("--commit-message", default="Import data from CSD3", help="Commit message for the ingestion")
    args = parser.parse_args()

    # Create the LakeFS manager instance
    lakefs_manager = LakeFSManager(repo_name=args.repo)

    branch = lakefs_manager.create_branch()
    lakefs_manager.upload_files_to_branch(branch.id, args.data_dir)
    lakefs_manager.commit_branch(branch.id, args.commit_message)
    lakefs_manager.show_diff(branch)

    # Merge the branch into main
    lakefs_manager.merge_branch(branch)
    lakefs_manager.delete_branch(branch.id)


if __name__ == "__main__":
    main()