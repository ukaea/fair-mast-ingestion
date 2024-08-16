from pendulum import datetime
import logging
from pathlib import Path
import sys
from airflow import DAG
sys.path.append("/home/ir-hods1/projects/fair-mast-ingestion")
from src.utils import read_shot_file
from src.workflow import CreateSignalMetadataTask, CreateSourceMetadataTask
from src.workflow import WorkflowManager, MetadataWorkflow
from airflow.decorators import dag, task, task_group

def signal_metadata_workflow(data_dir: str, shot: int):
    try:
        signal_metadata = CreateSignalMetadataTask(f"{data_dir}/signals", shot)
        signal_metadata()
    except Exception as e:
        logging.error(f"Could not parse signal metadata for shot {shot}: {e}")   

def source_metadata_workflow(data_dir: str, shot: int):
    try:
        source_metadata = CreateSourceMetadataTask(f"{data_dir}/source", shot)
        source_metadata()
    except Exception as e:
        logging.error(f"Could not parse source metadata for shot {shot}: {e}") 

@task
def run_workflows_serial_signal(data_dir, shot_list: list):
        n = len(shot_list)
        for i, shot in enumerate(shot_list):
            signal_metadata_workflow(data_dir, shot)
            logging.info(f"Done shot {i+1}/{n} = {(i+1)/n*100:.2f}%")
@task
def run_workflows_serial_source(data_dir, shot_list: list):
        n = len(shot_list)
        for i, shot in enumerate(shot_list):
            source_metadata_workflow(data_dir, shot)
            logging.info(f"Done shot {i+1}/{n} = {(i+1)/n*100:.2f}%")

default_args = {"start_date": datetime(2021, 1, 1)}
with DAG('metadata_processing', default_args=default_args, schedule_interval='@daily') as dag:
    @task
    def read_shots(shot_file):
        shot_list = read_shot_file(shot_file)
        return shot_list
    
    @task_group
    def metadata_workflow_group(data_dir, shot_list):
         run_workflows_serial_signal(data_dir, shot_list)
         run_workflows_serial_source(data_dir, shot_list)


    shots = read_shots(shot_file="/home/ir-hods1/projects/fair-mast-ingestion/campaign_shots/tiny_campaign.csv")
    metadata_workflow_group(data_dir="/home/ir-hods1/projects/fair-mast-ingestion/data/uda", shot_list=shots)