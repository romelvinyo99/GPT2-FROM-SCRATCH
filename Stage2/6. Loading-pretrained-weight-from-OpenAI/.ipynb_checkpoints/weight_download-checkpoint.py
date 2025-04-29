# Importations
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import requests
import json
import os


# Creating the loading function
def download_and_load_gpt2(model_dir, model_size):
    # Validating model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"model size not in {allowed_sizes}")
    # Defining the paths
    # model_dir + model_size = model_dir
    model_dir = os.path.join(model_dir, model_size)
    # The online file locations
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # Defining te file names
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]
    # Making the model_dir directory
    os.makedirs(model_dir, exist_ok=True)
    # Downloading the individual files
    for filename in filenames:
        # Defining the online file location = base url + model size + file name
        file_url = os.path.join(base_url, model_size, filename)
        # Defining the file path where to save the downloaded = model_dir + filename
        file_path = os.path.join(model_dir, filename)
        # Downloading the file
        download_files(file_url, file_path)
    # Extracting the settings and parameters
    # Getting the latest checkpoint path
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # Getting the settings - from the json file
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    # Getting the parameters
    params = load_gpt2_params_from_tf_cpkt_path(tf_ckpt_path, settings)
    return settings, params


def download_files(url, destination):
    try:
        # Sending a request to download the file disabling SSL Verification
        response = requests.get(url, stream=True, verify=False)
        # Getting the total file size from the response headers defaulting to zero
        file_size = int(response.headers.get("content-length", 0))
        # Check if file exists in the computer
        if os.path.exists(destination):
            # Getting the local file size
            file_size_local = os.path.getsize(destination)
            # Checking if the file size online and file size in computer is the same
            if file_size == file_size_local:
                print(f"File already exists and is upto date{destination}")
                return
                # Downloading the file in chunks
        chunk_size = 1024  # 1kb
        # Defining the progress bar
        progress_bar_description = url.split("/")[-1]
        with tqdm(total=file_size, desc=progress_bar_description, unit="iB", unit_scale=True) as progress_bar:
            # Opening the file in binary write mode
            with open(destination, "wb") as file:
                # Iterate file in chunks
                for chunk in response.iter_content(chunk_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        print("/033[41mError downloading the file/033m[0m")
        print(f"/033[41mCheck url: {url}/033[0m")


def load_gpt2_params_from_tf_cpkt_path(cpkt_path, settings):
    # Initialize the parameters dictionary - with blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    # Iterating over the variables in the ckpt path
    for (name, _) in tf.train.list_variables(cpkt_path):
        # Squeeze the tensor containing the values
        variable_array = np.squeeze(tf.train.load_variable(cpkt_path, name))
        # Extracting the import part names - removing the "model" prefix - "model/h1/attn/c_attn/bias"
        variable_name_parts = name.split("/")[1:]
        # Identifying the target dictionary
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            # Getting the layer number
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]
        # Recursively access/create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        # Getting the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array
    # Returning the final parameters dictionary
    return params

