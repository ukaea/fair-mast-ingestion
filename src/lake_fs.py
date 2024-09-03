import os
import glob
import config
from lakefs.client import Client
import lakefs
import subprocess
from datetime import datetime

class LakeFSManager:
    def __init__(self, host, username, password, repo_name):
        self.client = Client(host=host, username=username, password=password)
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
        # Generate a unique branch name based on the current date and time
        print("Creating branch for ingestion...")
        branch_name = datetime.now().strftime("branch-%Y%m%d-%H%M%S")
        branch = self.repo.branch(branch_name).create(source_reference="main")
        return branch

    def upload_files_to_branch(self, branch_name, local_folder="data/local"):
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

if __name__ == "__main__":
    manager = LakeFSManager(
        host="http://127.0.0.1:8000/",
        username=config.username,
        password=config.password,
        repo_name="example-repo"
    )

    # Create a new branch with a unique name
    branch = manager.create_branch()

    # Upload a local file to the new branch
    manager.upload_files_to_branch(
        branch_name=branch.id,
        local_folder="data"
    )

    # Commit the uploaded files to the branch
    manager.commit_branch(
        branch_name=branch.id,
        message="Uploaded new files."
    )

    # Show differences between the main branch and the new branch
    manager.show_diff(branch)

    # Merge the new branch into the main branch
    manager.merge_branch(branch)