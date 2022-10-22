from flask import Flask, render_template, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
import numpy as np
import os 
import matplotlib.pyplot as plt
from PIL import Image
import random, sys
from dataclasses import dataclass
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, PNDMScheduler
from torch import autocast
import math


# Checking for gpu and assigning to device
GPU_indx = 0
device = 'cpu'  # torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

# Create training configuration
@dataclass
class TrainingConfig:
    # Training hyperparameters
    image_size = 64
    eval_batch_size = 16
    mixed_precision = 'no'  # 'fp16'  # set as 'no' for float32
    model_dir = "models/SpriteModel"
    output_dir = "images/"

config = TrainingConfig()


# Inference
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, pipeline):
    with autocast("cuda"):
        images = pipeline(
            batch_size = config.eval_batch_size, 
            generator=torch.manual_seed(config.seed),
        )["sample"]

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)
    return images, image_grid

test_dir = config.output_dir
os.makedirs(test_dir, exist_ok=True)

model = UNet2DModel.from_pretrained(f'{config.model_dir}/unet', revision="fp32", torch_dtype=torch.float32).eval()

# Define noise scheduler
# List of noise schedulers: https://huggingface.co/docs/diffusers/api/schedulers 
noise_scheduler = PNDMScheduler(num_train_timesteps=1000)#, tensor_format="pt")
pipeline = PNDMPipeline(unet=model.to(device), scheduler=noise_scheduler).to(device)


# Initialize the flask app
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/get-image/<string:prompt>')
def get_image(prompt: str):
    image_path = generate_image(prompt)
    return send_file(image_path, mimetype='image/png')


def generate_image(prompt: str) -> str:
    seed = random.randrange(sys.maxsize)
    BATCH_SIZE = 100
    with autocast("cuda"):
        images = pipeline(
                batch_size = BATCH_SIZE,#config.eval_batch_size, 
                generator=torch.manual_seed(seed),
                )["sample"]
        
    image_grid = make_grid(images[0], rows=int(math.sqrt(BATCH_SIZE)), cols=int(math.sqrt(BATCH_SIZE)))
    image_filename = f"{test_dir}/{noise_scheduler._class_name}_{noise_scheduler.num_train_timesteps}ts_{seed}.png"
    image_grid.save(image_filename)
    
    return image_filename