'''
Contains the handler function that will be called by the serverless.
'''

from numpy import number
import runpod
import uuid

from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from PIL import Image

import io
import os
import sys
import platform
import fooocus_version

from modules.launch_util import is_installed, run, python, \
    run_pip, repo_dir, git_clone, requirements_met, script_path, dir_repos
from modules.model_loader import load_file_from_url
from modules.path import modelfile_path, lorafile_path, clip_vision_path, controlnet_path, vae_approx_path, fooocus_expansion_path


from PIL import Image

import time


from modules.resolutions import get_resolution_string
#ali oss auth
# https://help.aliyun.com/zh/oss/developer-reference/python-configuration-access-credentials



REINSTALL_ALL = False
# DEFAULT_ARGS = ['--disable-smart-memory']

INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'base_model_name': {
        'type': str,
        'required': False,
        'default': 'sd_xl_base_1.0_0.9vae.safetensors'
    },

}


REINSTALL_ALL = False


# all have been downloaded locally before build
def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.21')

    comfy_repo = os.environ.get('COMFY_REPO', "https://github.com/comfyanonymous/ComfyUI")
    comfy_commit_hash = os.environ.get('COMFY_COMMIT_HASH', "fb3b7282034a37dbed377055f843c9a9302fdd8c")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    comfyui_name = 'ComfyUI-from-StabilityAI-Official'
    # git_clone(comfy_repo, repo_dir(comfyui_name), "Inference Engine", comfy_commit_hash)
    sys.path.append(os.path.join(script_path, dir_repos, comfyui_name))

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if REINSTALL_ALL or not is_installed("xformers"):
        if platform.system() == "Windows":
            if platform.python_version().startswith("3.10"):
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
            else:
                print("Installation of xformers is not supported in this version of Python.")
                print(
                    "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                if not is_installed("xformers"):
                    exit(0)
        elif platform.system() == "Linux":
            run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


model_filenames = [
    ('sd_xl_base_1.0_0.9vae.safetensors',
     'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors'),
    ('sd_xl_refiner_1.0_0.9vae.safetensors',
     'https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors')
]

lora_filenames = [
    ('sd_xl_offset_example-lora_1.0.safetensors',
     'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors')
]

clip_vision_filenames = [
    ('clip_vision_g.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors')
]

controlnet_filenames = [
    ('control-lora-canny-rank128.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-canny-rank128.safetensors'),
    ('control-lora-canny-rank256.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors'),
    ('control-lora-depth-rank128.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors'),
    ('control-lora-depth-rank256.safetensors',
     'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors')
]

vae_approx_filenames = [
    ('taesdxl_decoder.pth',
     'https://huggingface.co/lllyasviel/misc/resolve/main/taesdxl_decoder.pth')
]

def download_models():
    for file_name, url in model_filenames:
        load_file_from_url(url=url, model_dir=modelfile_path, file_name=file_name)
    for file_name, url in lora_filenames:
        load_file_from_url(url=url, model_dir=lorafile_path, file_name=file_name)
    for file_name, url in clip_vision_filenames:
        load_file_from_url(url=url, model_dir=clip_vision_path, file_name=file_name)
    for file_name, url in controlnet_filenames:
        load_file_from_url(url=url, model_dir=controlnet_path, file_name=file_name)
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=vae_approx_path, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=fooocus_expansion_path,
        file_name='pytorch_model.bin'
    )

    return


def clear_comfy_args():
    argv = sys.argv
    sys.argv = [sys.argv[0]]
    import comfy.cli_args
    sys.argv = argv

# def cuda_malloc():
#     import cuda_malloc


prepare_environment()

clear_comfy_args()
# cuda_malloc()


download_models()


import modules.async_worker as worker


def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # print("Start generating...")
    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}

    # Extracting the new parameters
    prompt = validated_input['validated_input'].get('prompt')


    base_model_name = validated_input['validated_input'].get('base_model_name')


    task = (prompt, '', [], 'Speed', "1024×1024", 1, 'None',
        2, 'dpmpp_2m_sde_gpu', 'karras', 24, 0.75, 7,
        base_model_name, 'sd_xl_refiner_1.0_0.9vae.safetensors', -2, -2,
        'None', 0.5,'None', 0.5,'None', 0.5,'None', 0.5,'None', 0.5, False, False,
        False, 0.06, 0.8,
        False, 1, 1, 1, 1,
        1, 1, False, 'png',
        False, 0.2, 0.8, 0, 0.4, 0.5, 'control-lora-canny-rank128.safetensors',
        False, 0, 0.4, 0.5, 'control-lora-depth-rank128.safetensors', True,
        [], [], False)


    worker.buffer.append(list(task))

    finished = False

    result = []
    metadata = []
    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == 'preview':
                pass
            if flag == 'results':
                result = product

            if flag == 'metadatas':
                finished = True
                metadata = product

    print(f"Generate meta : {metadata}")

    if len(result) == 0:
        raise ValueError('Failed to generate image')

    return metadata

runpod.serverless.start({"handler": generate_image})
