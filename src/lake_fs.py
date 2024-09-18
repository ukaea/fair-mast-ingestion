import os
import argparse
import logging
from lakefs.client import Client
from src.task import CleanupDatasetTask
import lakefs
import subprocess
from datetime import datetime
import uuid
import sys

def execute_command(command):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info("Command output: %s", result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            logging.error("Error executing command: %s", e.stderr)
            return e
    
def create_branch(repo):
        logging.info("Uploading files from data directory to branch...")
        branch_name = f"branch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        command = [
            "lakectl", "branch", "create",
            f"lakefs://{repo}/{branch_name}/",
            "-s", f"lakefs://{repo}/main/"
        ]
        execute_command(command)
        return branch_name
        
def lakefs_merge_into_main(repo, branch):
        logging.info("Uploading files from data directory to branch...")
        command = [
            "lakectl", "merge",
            f"lakefs://{repo}/{branch}/",
            f"lakefs://{repo}/main/",
            "-m", f"Merge {branch} branch into main"
        ]
        execute_command(command)