import comet_ml
from comet_ml.integration.pytorch import watch

comet_ml.login(project_name="fewshotadapter")



from datetime import datetime
import os
import sys
os.chdir('/home/dcor/niskhizov/AdversarialRendering/mitsuba/gsoup/src')
sys.path.insert(0, '/home/dcor/niskhizov/AdversarialRendering/mitsuba/gsoup/src')

import glob
import mitsuba as mi
import drjit as dr
import numpy as np
import cv2
from gsoup.image import (
    change_brightness,
    add_alpha,
    resize,
    tonemap_reinhard,
    linear_to_srgb,
)
import matplotlib.pyplot as plt
import pickle

mi.set_variant(
    "cuda_ad_rgb"
    # "scalar_rgb"
)  # "llvm_ad_rgb", "scalar_rgb" # must set before defining the emitter


patchs = glob.glob('/home/dcor/niskhizov/AdversarialRendering/botorch_snapshots/*patch*')

anchors_dir = '/home/dcor/niskhizov/rgb_patterns_yotam/'
anchors_paths = glob.glob(f'{anchors_dir}/*')

cam_wh=(256, 256)
cam_fov=45
proj_wh=(256, 256)
proj_fov=45
ambient_color=list(np.array([0.1, 0.1, 0.1] )*10)
proj_brightness = 30

spp=256
proj_response_mode = "srgb"

import gsoup
from gsoup.projector_plugin_mitsuba import ProjectorPy
import mitsuba as mi
mi.register_emitter("projector_py", lambda props: ProjectorPy(props))

# roughness = list(range(0.1,0.8, 0.1))
# metallics  = list(range(0, 1.0, 0.1))

roughness = list(np.arange(0.1, 0.8, 0.1))

metallics  = list(np.arange(0, 1.0, 0.1))

import torch
import torch.nn.functional as F




def create_gaussian_kernel_2d(kernel_size, sigma):
    """Create a 2D Gaussian kernel using PyTorch"""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Create 2D kernel
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel_2d.unsqueeze(0).unsqueeze(0)

def gaussian_glow(image, glow_radius, glow_strength):
    """Apply Gaussian blur using PyTorch convolution with groups (no for loops)"""
    # Ensure kernel size is odd
    kernel_size = glow_radius

    if kernel_size % 2 == 0:
        kernel_size += 1
    

    sigma = glow_radius / 6.0  # Adjust sigma based on radius
    # Create Gaussian kernel
    kernel = create_gaussian_kernel_2d(kernel_size, sigma).to(image.device)
    
    # Reshape image from (H, W, C) to (1, C, H, W) for conv2d
    img_4d = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # Repeat kernel for each channel: (3, 1, kernel_size, kernel_size)
    kernel_3ch = kernel.repeat(3, 1, 1, 1)
    
    # Apply convolution with groups=3 (one kernel per channel)
    padding = kernel_size // 2
    blurred_4d = F.conv2d(img_4d, kernel_3ch, padding=padding, groups=3)
    
    # Convert back to (H, W, C) format
    img_blurred = blurred_4d.squeeze(0).permute(1, 2, 0)

    img_blended = image + img_blurred * glow_strength

    
    return img_blended


def gaussian_blur_torch(image, sigma):

    device = image.device if hasattr(image, 'device') else 'cpu'

    # Create Gaussian kernel
    kernel_size = int(2 * sigma * 3) + 1  # 6 sigma rule
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel - ensure it's on the same device
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Create 2D kernel
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)


    gauss_2d_rgb = gauss_2d.repeat(3, 1, 1, 1)  # Repeat for RGB channels
    
    # Apply convolution with padding
    padding = kernel_size // 2
    blurred = F.conv2d(image, gauss_2d_rgb, padding=padding,groups=3)
    
    return blurred.squeeze(0).squeeze(0)


def generate_random_scene(test=False):
    """
    Generate a random scene dictionary for Mitsuba.
    """            
    if test:
        random_patch = np.random.choice(patchs[-100:])
    else:
        random_patch = np.random.choice(patchs[:-100])
    roughnes =  np.random.choice(roughness)
    metallic = np.random.choice(metallics)

    proj_brightness = np.random.uniform(1.0, 10.0)

    ambient_color_scale = np.random.uniform(0.1, 1.0)

    ambient_color=list(np.array([1.0,1.0,1.0] )*ambient_color_scale)

    focus_distance = np.random.uniform(1.0, 20.0)

    random_patch_rgb = cv2.cvtColor(cv2.imread(random_patch), cv2.COLOR_BGR2RGB)#gsoup.generate_voronoi_diagram(512, 512, 1000)*0+255
    random_patch_rgb = gsoup.to_float(random_patch_rgb)
    
    # random_patch_rgb  =  random_patch_rgb.clip(0.4,0.7)
    random_patch_rgb = (random_patch_rgb - random_patch_rgb.min()) / (random_patch_rgb.max() - random_patch_rgb.min())

    random_patch_rgb = random_patch_rgb*0.6 + 0.2

    scene_dict = {

                    "type": "scene",

                    "integrator": {
                        "type": "path",
                        "hide_emitters": True,
                        "max_depth": 6,
                    },
                    "camera": {
                        "type": "thinlens",
                        'aperture_radius': 0.02,  # control DoF (0.0 = pinhole)
                        'focus_distance': focus_distance,  # meters to subject
                        "fov": cam_fov,
                        
                        "to_world": mi.ScalarTransform4f().look_at(
                            origin=[0.1, 0, 0.0],  # along +X axis
                            target=[0, 0, 0],
                            up=[0, 0, 1],  # Z-up
                        ),
                        "film": {
                            "type": "hdrfilm",
                            "width": cam_wh[0],
                            "height": cam_wh[1],
                            "pixel_format": "rgba",
                            "rfilter": {"type": "box"},
                        },
                        "sampler": {
                            "type": "independent",
                            "sample_count": spp,  # number of samples per pixel
                        },
                    },

                    "wall3": {
                        "type": "rectangle",
                        "to_world": mi.ScalarTransform4f()
                        .translate([-2.0, 0.0, 0.0])
                        .rotate([0, 1, 0], 90),
                        "bsdf": 
                        {

                        'type': 'principled',
                        'base_color': {
                            'type': 'bitmap',
                            # 'filename': f'{random_patch}',
                            'data' : random_patch_rgb,

                        },
                        'roughness' : roughnes,
                        'metallic': metallic,
                        },
                    },

                    "projector": {
                        "type": "projector_py",
                        "irradiance": {
                            # "type": "ref",
                            # "id": "proj_texture",
                            "type": "bitmap",
                            'filename': '/home/dcor/niskhizov/AdversarialRendering/mitsuba/mat_data/WoodFloor064_1K/WoodFloor064_1K-JPG_Color.jpg',
                            # "raw": True,  # assuming the image is in linear RGB
                            'format' : 'variant'
                        },
                        "scale": proj_brightness,
                        "to_world": mi.ScalarTransform4f().look_at(
                            origin=[0.005, 0, 0],  # along +X axis
                            target=[0, 0, 0],
                            up=[0, 0, 1],  # Z-up
                        ),
                        "fov": proj_fov,
                        "response_mode": proj_response_mode,
                    },
                    
                    "ambient": {
                        "type": "constant",
                        "radiance": {"type": "rgb", "value": ambient_color},
                    },
                }
    
    return scene_dict
# params = mi.traverse(scene)

import torch



@dr.wrap(source='torch', target='drjit')
def render_texture2(params, proj_tex_grad):



    params['projector.irradiance.data'] = proj_tex_grad 
    params.update()

    raw_render = mi.render(scene, params)



    return raw_render

def render(params, proj_tex_grad):

    scene_params = params['scene']
    cam_params = params['cam_params']

    raw_render = render_texture2(scene_params,proj_tex_grad)


    alpha = raw_render[:, :, -1:]
    image = raw_render[:, :, :3]
    # no_alpha_render = gsoup.alpha_compose(render)
    image = tonemap_reinhard(image, exposure=1.0)
    # image = linear_to_srgb(image)
    final_image = add_alpha(image, alpha)

    final_image  = gaussian_blur_torch(final_image[:, :, :3].permute(2,0,1), cam_params['sigma']).permute(1,2,0)

    final_image = gaussian_glow(final_image, cam_params['glow_radius'], cam_params['glow_strength'])

    final_image = final_image.clamp(0, 1.0)

    return final_image





texture = cv2.imread('/home/dcor/niskhizov/AdversarialRendering/mitsuba/colorful_pattern.jpg')#gsoup.generate_voronoi_diagram(512, 512, 1000)*0+255
proj_texture = gsoup.to_float(texture)

scene_dict = generate_random_scene()

scene = mi.load_dict(scene_dict)


params = mi.traverse(scene)

# cam_params = {'sigma': 1.0}  # Example camera parameters, adjust as needed
# params = {'scene': params, 'cam_params': cam_params}

# proj_tex_grad = torch.from_numpy(proj_texture).requires_grad_(True).cuda()  # Convert to (C, H, W) format for PyTorch

# final_image = render(params, proj_tex_grad)

# plt.imshow(final_image.detach().cpu().numpy(), aspect='auto')
import torchvision

resizer = torchvision.transforms.Resize((256, 256))
# num_anchors = 10
# learnable_anchors = torch.rand(num_anchors, 3, 3, 3).cuda()  # Example learnable anchors
# learnable_anchors = learnable_anchors.requires_grad_(True)

proj_tex = torch.from_numpy(proj_texture).cuda() 
proj_tex_r = resizer(proj_tex.permute(2, 0, 1))

anchors = []#torch.rand(num_anchors, 3, 10, 10).cuda()  # Example learnable anchors
for i in range(len(anchors_paths)):
    anchor = cv2.imread(anchors_paths[i])
    anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
    anchor = gsoup.to_float(anchor)
    anchor = torch.from_numpy(anchor).permute(2, 0, 1).cuda()  # Convert to (C, H, W) format
    anchors.append(anchor)
anchors = torch.stack(anchors).cuda()  # Stack anchors into a single tensor
    
num_anchors = anchors.shape[0]



class AnchorsCnnDownsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AnchorsCnnDownsample, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(num_anchors*16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )


        # self.fc =  torch.nn.Sequential(
        #     torch.nn.Linear(8 * 16 * 16, 32),  # Assuming input size is 400x400
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(32, 32),  
        #     torch.nn.ReLU(),
        # )

        # self.fc2 = torch.nn.Sequential(torch.nn.Linear(32*learnable_anchors.shape[0], 1024),
        #                                 torch.nn.ReLU(),
        #                                 torch.nn.Linear(1024, 1024),
        #                                 torch.nn.ReLU(),
                                       
                                       
        #                                )

    def forward(self, x):
        # print(x.shape)
        x = self.layers(x)#.reshape(3*learnable_anchors.shape[0], 256, 256))  
        # print(x.shape)
        x = self.deconv(x.reshape(num_anchors*16, 16, 16))  
        # print(x.shape)
        # x = x.reshape(x.size(0), -1)  # Flatten the tensor
        # x = self.fc(x)
        # x = x.reshape(-1)
        # x = self.fc2(x)
        return x
    

import torch.nn as nn

class LightCNN(nn.Module):
    def __init__(self):
        super(LightCNN, self).__init__()
        self.im_encoder = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 4x60x80 -> 8x60x80
        self.anchor_encoder = nn.Conv2d(num_anchors*3, num_anchors, kernel_size=3, padding=1, groups=num_anchors)  # 4x60x80 -> 8x60x80
        self.encoder = nn.Sequential(
            nn.Conv2d(32+num_anchors, 32, kernel_size=3, padding=1),  # 4x60x80 -> 8x60x80
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 8x60x80 -> 16x60x80
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 8x60x80 -> 16x60x800
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 8x60x80 -> 16x60x80
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1), # 16x60x80 -> 8x60x80
            nn.LeakyReLU(),
            nn.Conv2d(8, 3, kernel_size=3, padding=1) )  # 8x60x80 -> 4x60x80
        


    def forward(self, x, condition):
        im_enc = self.im_encoder(x)
        anchor_enc = self.anchor_encoder(condition)
        x = torch.cat([im_enc, anchor_enc], dim=0)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


ckpt_dir = 'v9_fixed_ckpt'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
# cnn_ds = AnchorsCnnDownsample(3, 64).cuda()
model = LightCNN().cuda()  

if os.path.exists(f'{ckpt_dir}/light_cnn.pth') :
    print('Loading existing models...')
    # learnable_anchors = torch.load(f'{ckpt_dir}/learnable_anchors.pth').cuda()
    light_cnn_sd = torch.load(f'{ckpt_dir}/light_cnn.pth')
    # cnn_downsample_sd = torch.load(f'{ckpt_dir}/cnn_downsample.pth')
    # cnn_ds.load_state_dict(cnn_downsample_sd)
    model.load_state_dict(light_cnn_sd)
else:
    print('No existing models found, initializing new ones...')
    

from torch import optim

  



optimizer = optim.Adam([*model.parameters()], lr=1e-3)
# freeze the learnable anchors
# learnable_anchors = learnable_anchors.requires_grad_(False)
# optimizer = optim.Adam([*cnn_ds.parameters(), *model.parameters()], lr=1e-3)



mi.set_log_level(mi.LogLevel.Error)  # Suppress Mitsuba logs
import tqdm
proj_tex_r.shape
from torchmetrics.image import TotalVariation
tv = TotalVariation().cuda()
caltec_photos = '/home/dcor/niskhizov/caltech-101/101_ObjectCategories/'
ls = glob.glob(caltec_photos + '/*/*.jpg')

start_time = str(datetime.now()).split('.')[0]

experiment = comet_ml.Experiment(display_summary_level=0, log_code=True)

anchor_size = anchors.shape[2]
exp_name = f'{num_anchors}_anchors_{anchor_size}x{anchor_size}_16c_condition'

complete_name = exp_name+'_'+start_time

experiment.set_name(complete_name)
experiment.log_code()

attempts_per_scene = 10

experiment.log_parameters({
    'num_anchors': num_anchors,
    'anchor_size': anchors.shape[2],
    'anchor_channels': anchors.shape[1],
    'cam_wh': cam_wh,
    'cam_fov': cam_fov,
    'proj_wh': proj_wh,
    'proj_fov': proj_fov,
    'spp': spp,
    'proj_response_mode': proj_response_mode,
    'exp_name': complete_name,
    'ckpt_dir': ckpt_dir,
    })
watch(model)
# watch(cnn_ds)

L1 = nn.L1Loss()


def generate_cam_params():
    sigma = np.random.uniform(1, 4)  # Random sigma for Gaussian blur
    glow_radius = np.random.randint(10, 20)  # Random glow radius
    glow_strength = np.random.uniform(1.0, 2.0)  # Random glow strength

    return {'sigma':sigma, 'glow_radius': glow_radius, 'glow_strength': glow_strength}

for epoch in tqdm.tqdm(range(10000)):

    cam_params = generate_cam_params()
    scene_dict = generate_random_scene()
    scene = mi.load_dict(scene_dict)
    params = {'scene':mi.traverse(scene), 'cam_params': cam_params}

    optimizer.zero_grad()

    # Resize the learnable anchors to match the input size
    projected_anchors = []

    for anchor in anchors:
        resized_anchor = resizer(anchor)  # Change to HWC format
        projected_anchors.append(render(params, resized_anchor.permute(1,2,0)))
    
    projected_anchors = torch.stack(projected_anchors) 
    # raise

    condition = projected_anchors.permute(0,3,1,2)#cnn_ds(projected_anchors.permute(0,3,1,2))
    condition = condition.reshape(-1,condition.shape[2], condition.shape[3])

    outputs = []
    true_projs = []
    for attempt in range(attempts_per_scene):
        # Randomly select a Caltech photo for texture
        random_caltech_photo = np.random.choice(ls[:-100])
        texture = cv2.cvtColor(cv2.imread(random_caltech_photo), cv2.COLOR_BGR2RGB)#gsoup.generate_voronoi_diagram(512, 512, 1000)*0+255
        proj_texture = gsoup.to_float(texture)
        proj_tex = torch.from_numpy(proj_texture).cuda() 
        proj_tex_r = resizer(proj_tex.permute(2, 0, 1))

        true_proj = render(params, proj_tex_r.permute(1,2,0))

        # model_input = torch.cat([proj_tex_r, condition])
        output = model(proj_tex_r,condition)

        outputs.append(output)
        true_projs.append(true_proj.permute(2,0,1))  # Ensure
    

    outputs = torch.stack(outputs)  # Shape: (attempts_per_scene, C, H, W)
    true_projs = torch.stack(true_projs)  # Shape: (attempts_per_scene, C, H, W)
    
    tv_loss = tv(anchors) / 10000
    
    loss = L1(outputs, true_projs) # + tv_loss# Ensure true_proj is in (C, H, W) format
    
    loss.backward()

    # clip gradients to avoid exploding gradients
    # learnable_anchors_grad = learnable_anchors.grad
    # cnn_ds_grad = torch.nn.utils.clip_grad_norm_(cnn_ds.parameters(), max_norm=10.0)
    model_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    

    optimizer.step()

    if epoch % 10 == 0:
        experiment.log_metrics({
            "loss": loss.item(),
            "tv_loss": tv_loss.item(),
            # "learnable_anchors_grad_mean": learnable_anchors_grad.mean().item(),
            # "cnn_ds_grad": cnn_ds_grad,
            "model_grad": model_grad,
        }, step=epoch)

        
    if (epoch) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        plt.imshow(output.permute(1,2,0).detach().cpu().numpy(), aspect='auto')
        plt.show()
        plt.imshow(true_proj.detach().cpu().numpy(), aspect='auto')
        plt.show()

        # Save the model state
        # torch.save(learnable_anchors, f'{ckpt_dir}/learnable_anchors.pth')
        torch.save(model.state_dict(), f'{ckpt_dir}/light_cnn.pth')
        # torch.save(cnn_ds.state_dict(), f'{ckpt_dir}/cnn_downsample.pth')
        # Save the current learnable anchors
        with torch.no_grad():
            
            scene_dict = generate_random_scene(test=True)
            scene = mi.load_dict(scene_dict)
            params = mi.traverse(scene)         
            cam_params = generate_cam_params()
            params = {'scene': params, 'cam_params': cam_params}

            # Resize the learnable anchors to match the input size
            projected_anchors = []

            for anchor in anchors:
                resized_anchor = resizer(anchor)  # Change to HWC format
                projected_anchors.append(render(params, resized_anchor.permute(1,2,0)))

            projected_anchors = torch.stack(projected_anchors) 
            # raise

            condition = projected_anchors.permute(0,3,1,2)#cnn_ds(projected_anchors.permute(0,3,1,2))
            condition = condition.reshape(-1,condition.shape[2], condition.shape[3])

            random_caltech_photo = np.random.choice(ls[:-100])
            texture = cv2.cvtColor(cv2.imread(random_caltech_photo), cv2.COLOR_BGR2RGB)#gsoup.generate_voronoi_diagram(512, 512, 1000)*0+255
            proj_texture = gsoup.to_float(texture)
            proj_tex = torch.from_numpy(proj_texture).cuda() 
            proj_tex_r = resizer(proj_tex.permute(2, 0, 1))

            true_proj = render(params, proj_tex_r.permute(1,2,0))

            # model_input = torch.cat([proj_tex_r, condition])
            output = model(proj_tex_r,condition)

            # physical test 
            with open("/home/dcor/niskhizov/AdversarialRendering/mitsuba/yotam_patterns_jeep_condition.pkl",'rb') as f:
                condition_jeep_yotam = pickle.load(f).cuda()

            with open("/home/dcor/niskhizov/AdversarialRendering/mitsuba/yotam_patterns_jeep_frame_unwarped.pkl",'rb') as f:
                jeep_frame_unwarped = pickle.load(f).cuda()

            output_jeep = model(jeep_frame_unwarped, condition_jeep_yotam)

        physical_loss = L1(output_jeep, jeep_frame_unwarped)
        experiment.log_metrics({
            "physical_loss": physical_loss.item(),
        }, step=epoch)
        experiment.log_image(output_jeep.permute(1,2,0).detach().cpu().numpy(), name=f'output_jeep.png', step=epoch)
        experiment.log_image(jeep_frame_unwarped.permute(1,2,0).detach().cpu().numpy(), name=f'jeep_frame_unwarped.png', step=epoch)
        experiment.log_image(output.permute(1,2,0).detach().cpu().numpy(), name=f'output.png', step=epoch)
        experiment.log_image(true_proj.detach().cpu().numpy(), name=f'true_proj.png', step=epoch)
        experiment.log_image(proj_tex_r.permute(1,2,0).detach().cpu().numpy(), name=f'proj_tex_r.png', step=epoch)
        experiment.log_image(texture, name=f'texture.png', step=epoch)
        

        # for i in range(learnable_anchors.shape[0]):
        #     experiment.log_image(learnable_anchors[i].cpu().detach().numpy().transpose(1, 2, 0), name=f'learnable_anchor_{i}.png', step=epoch)

        
                