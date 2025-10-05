
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=plastic&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-ee4c2c?style=plastic&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-latest-9C27B0?style=plastic&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/)
[![fastai](https://img.shields.io/badge/fastai-latest-009688?style=plastic&logo=fastai&logoColor=white)](https://docs.fast.ai/)
[![fastcore](https://img.shields.io/badge/fastcore-latest-607d8b?style=plastic)](https://github.com/fastai/fastcore)
[![python-box](https://img.shields.io/badge/python--box-latest-4caf50?style=plastic)](https://github.com/cdgriffith/Box)
[![tqdm](https://img.shields.io/badge/tqdm-latest-ff9800?style=plastic&logo=python&logoColor=white)](https://tqdm.github.io/)
[![Pillow](https://img.shields.io/badge/Pillow-latest-3f51b5?style=plastic&logo=python&logoColor=white)](https://python-pillow.org/)
[![uvicorn](https://img.shields.io/badge/uvicorn-latest-00bcd4?style=plastic)](https://www.uvicorn.org/)
[![dvc-gdrive](https://img.shields.io/badge/dvc--gdrive-latest-795548?style=plastic&logo=google-drive&logoColor=white)](https://dvc.org/doc/user-guide/setup-google-drive-remote)
# Magnetic Tile Surface Defect Detection
======================================

This project focuses on detecting surface defects in magnetic tiles using an image segmentation model trained with DVC pipelines and a FastAPI-based web API for real-time inference. The dataset used is the [Magnetic Tile Defect Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets).

Project Overview
----------------

The project consists of two main components:

1.  **Image Segmentation Model Training**:

    -   Utilizes [DVC](https://dvc.org/) pipelines to manage data loading, splitting, model training, and evaluation.

    -   Trains a U-Net model with a ResNet34 backbone using the FastAI library for binary segmentation (defect vs. defect-free).

    -   Processes the Magnetic Tile Defect Dataset, which includes images and corresponding binary masks.

2.  **Web API**:

    -   Built with [FastAPI](https://fastapi.tiangolo.com/) to provide an endpoint (`/analyze`) for uploading an image and receiving a binary segmentation mask overlaid on the original image.

    -   Returns base64-encoded images: the original (cropped) image and the image with a semi-transparent red mask highlighting defective areas.

Dataset
-------

The dataset is sourced from the [Magnetic Tile Defect Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets). It contains images of magnetic tiles with various surface defects (e.g., blowholes, cracks, frays) and corresponding binary masks for segmentation tasks.

Project Structure
-----------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plaintext
├── data/                           # Directory for dataset and processed data
├── models/                         # Directory for trained model artifacts
├── app/ ├── main.py                # end point
├── src/                            # Source code for DVC pipelines
│   ├── stages/                     # Pipeline stages (data loading, splitting, training, evaluation)
│   │   ├── data_load.py
│   │   ├── data_split.py
│   │   ├── train.py
│   │   ├── eval.py
│   ├── utils/                      # Utility scripts for data processing and training
│   │   ├── data_utils.py
│   │   ├── eval_utils.py
│   │   ├── load_params.py
│   │   ├── train_utils.py
├── training/                       # Directory for training metrics and plots
├── evaluation/                     # Directory for evaluation metrics
├── dvc.yaml                        # DVC pipeline configuration
├── params.yaml                     # Configuration parameters for the pipeline
                      # FastAPI web server for inference
└── README.md                       # Project documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setup Instructions
------------------

### Prerequisites

-   Python 3.8+

-   [DVC](https://dvc.org/doc/install)

-   [FastAI](https://docs.fast.ai/)

-   [PyTorch](https://pytorch.org/)

-   [FastAPI](https://fastapi.tiangolo.com/)

-   [Uvicorn](https://www.uvicorn.org/)

-   [dvclive](https://dvc.org/doc/dvclive)

-   Other dependencies listed in `pkg_list.txt`

### Installation

1.  Clone the repository:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    git clone <repository-url>
    cd <repository-name>
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.  Install dependencies:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    pip install -r pkg_list.txt
    pip install fastapi uvicorn
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.  Install DVC:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    pip install dvc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.  Pull the dataset and model artifacts (if available):

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    dvc pull
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Running the DVC Pipeline locally 
1.  Run the entire pipeline to train the model:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    dvc repro
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.  The pipeline includes the following stages:

    -   **data_load**: Downloads and preprocesses the dataset, organizing images and masks.

    -   **data_split**: Splits the dataset into training and test sets (default: 80% train, 20% test).

    -   **train**: Trains the U-Net model and saves it as `models/model_pickle_fastai.pkl`.

    -   **evaluate**: Evaluates the model on the test set, computing metrics (Dice, Jaccard, accuracy) and saving predictions.

3.  View metrics and plots:

    -   Metrics: `training/metrics.json`, `evaluation/metrics.json`

    -   Plots: `training/plots`

### Running the Web API

1.  Ensure the trained model (`models/model_pickle_fastai.pkl`) is available in the `models/` directory.

2.  Start the FastAPI server:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.  Access the API:

    -   Endpoint: `POST /analyze`

    -   Request: Upload an image file (e.g., JPG, PNG).

    -   Response: JSON with two base64-encoded images:

        -   `original`: The cropped input image.

        -   `overlay_result`: The input image with a semi-transparent red mask over defective areas.

    Example using `curl`:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    curl -X POST "http://localhost:8000/analyze" -F "image=@path/to/image.jpg"
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Example using Python `requests`:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ python
    import requests

    with open("path/to/image.jpg", "rb") as f:
        response = requests.post("http://localhost:8000/analyze", files={"image": f})
    result = response.json()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration
-------------

The pipeline is configured via `params.yaml`. Key parameters include: - **data_load**: Dataset URL, directories for images and masks. - **data_split**: Test set percentage, paths for train/test splits. - **train**: Image size (default: 224x224), learning rate, batch size, number of epochs, augmentation settings. - **evaluate**: Options for saving test predictions and metrics file path.

Evaluation Metrics
------------------

The evaluation stage computes: - **Dice Coefficient**: Measures overlap between predicted and true masks. - **Jaccard Coefficient (IoU)**: Measures intersection over union. - **Accuracy**: Pixel-wise accuracy across test images.

Metrics are saved in `evaluation/metrics.json`.

Web API Details
---------------

The web API (`main.py`) is built with FastAPI and includes: - **Endpoint**: `/analyze` - **Input**: An image file (processed to 224x224 resolution). - **Output**: A JSON response containing: - `original`: Base64-encoded cropped input image. - `overlay_result`: Base64-encoded image with a red semi-transparent mask (RGBA: 255, 0, 0, 120) over defective areas. - **CORS**: Enabled to allow requests from any origin. - **Model Loading**: Loads the trained model (`model_pickle_fastai.pkl`) at startup and uses CPU for inference.

To test the API locally, you can use tools like Postman or a simple HTML form:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ html
<form action="http://localhost:8000/analyze" method="post" enctype="multipart/form-data">
    <input type="file" name="image">
    <input type="submit">
</form>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage
-----

1.  **Training**:

    -   Run `dvc repro` to execute the full pipeline.

    -   Monitor training progress via `training/metrics.json` and `training/plots`.

2.  **Inference**:

    -   Start the FastAPI server (`uvicorn main:app --host 0.0.0.0 --port 8000`).

    -   Upload an image to the `/analyze` endpoint to receive the segmentation mask.

    -   Alternatively, use `eval.py` to evaluate the model on the test set.



### Setting Up Google Drive for DVC Remote Storage

To store datasets and model artifacts on Google Drive, follow these steps:

#### Create Google Drive Folder

1.  Go to [Google Drive](https://drive.google.com) and create a new folder for your project data.

2.  Open the folder and copy the folder ID from the URL (the string after `/folders/` and before the next parameter).

3.  Use underscores (`_`) instead of spaces in folder names to avoid path issues.

#### Connect Git Repository

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
git remote add origin https://github.com/yourusername/your-repo.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### Configure DVC Remote

Add Google Drive as the default remote storage:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
dvc remote add -d mydrive gdrive://YOUR_FOLDER_ID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace `YOUR_FOLDER_ID` with your actual Google Drive folder ID and `mydrive` with your chosen remote name.

#### Google Cloud Setup for Personal Drive

For personal Google Drive accounts, additional setup is required:

1.  **Create Google Cloud Project**:

    -   Go to [Google Cloud Console](https://console.cloud.google.com/).

    -   Create a new project dedicated to DVC remote access.

2.  **Enable Google Drive API**:

    -   In the project, navigate to **APIs & Services → Library**.

    -   Search for **Google Drive API** and enable it.

3.  **Set Up OAuth Consent Screen**:

    -   Go to **APIs & Services → OAuth consent screen**.

    -   Choose **External** user type (if not using Google Workspace).

    -   Select **Publish app** to avoid verification issues.

    -   Fill required fields (e.g., App name: "Data Version Control · DVC", logo, scopes).

    -   Add required Drive scopes (e.g., `drive.file` and others DVC requires).

    -   Note: Google may require app verification for sensitive/restricted scopes.

4.  **Create OAuth Credentials**:

    -   Go to **Credentials → Create Credentials → OAuth client ID**.

    -   Choose **Desktop App** for local development or **Web App** as appropriate.

    -   After creation, download the credentials file containing Client ID and Client Secret.

5.  **Configure DVC Authentication**:

    -   Set up authentication using the OAuth credentials.

    -   Configure DVC to use the Google Drive remote:

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
        dvc remote default mydrive
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

6.  **Verification**:

    -   Test the connection:

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
        dvc push
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    -   This should upload your DVC-tracked data to the configured Google Drive remote.

#### Important Notes


-   For personal Google Drive accounts, the service account method requires the Google Cloud setup.

-   Publishing the OAuth app helps avoid verification delays.


#### Troubleshooting


-   Verify the Google Drive API is enabled in your Google Cloud project.

-   Check that OAuth scopes include necessary Drive permissions.

### Running the DVC Pipeline

1.  Run the entire pipeline to train the model:

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
    dvc repro
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.  The pipeline includes the following stages:

    -   **data_load**: Downloads and preprocesses the dataset, organizing images and masks.

    -   **data_split**: Splits the dataset into training and test sets (default: 80% train, 20% test).

    -   **train**: Trains the U-Net model and saves it as `models/model_pickle_fastai.pkl`.

    -   **evaluate**: Evaluates the model on the test set, computing metrics (Dice, Jaccard, accuracy) and saving predictions.


Contributing
------------

Contributions are welcome! Please submit a pull request or open an issue for bugs, feature requests, or improvements.

License
-------

This project is licensed under the MIT License.
