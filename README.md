# Drug Prediction
This package is code to deploy a Sagemaker endpoint. The trained models are based off of previously run Sagemaker training jobs

# Build Model and Train
Sagemaker endpoints utilize docker containers stored in AWS Elastic Container Registry to execute. Code for deploying the container based inference jobs can be found in the <b> [run_container.ipynb](./run_container.ipynb) </b>

Steps for running a training job are as follows:
1. Build container and upload to AWS ECR using the build_and_push.sh script <br />
    `!cd container; ./build_and_push.sh gnn-inference`
3. Refer to [run_container.ipynb](./run_container.ipynb) on deploying Sagemaker endpoint and running inference

