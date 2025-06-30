
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

mi.set_variant(
    "cuda_ad_rgb"
    # "scalar_rgb"
)  # "llvm_ad_rgb", "scalar_rgb" # must set before defining the emitter


patchs = glob.glob('/home/dcor/niskhizov/AdversarialRendering/botorch_snapshots/*patch*')

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


def generate_random_scene():
    """
    Generate a random scene dictionary for Mitsuba.
    """            
    random_patch = np.random.choice(patchs)
    roughnes =  np.random.choice(roughness)
    metallic = np.random.choice(metallics)

    proj_brightness = np.random.uniform(1.0, 10.0)

    ambient_color_scale = np.random.uniform(0.1, 1.0)

    ambient_color=list(np.array([1.0,1.0,1.0] )*ambient_color_scale)

    focus_distance = np.random.uniform(1.0, 20.0)

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
                            'filename': f'{random_patch}',

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

    raw_render = render_texture2(params,proj_tex_grad)


    alpha = raw_render[:, :, -1:]
    image = raw_render[:, :, :3]
    # no_alpha_render = gsoup.alpha_compose(render)
    image = tonemap_reinhard(image, exposure=1.0)
    # image = linear_to_srgb(image)
    final_image = add_alpha(image, alpha)

    return final_image[:,:,:3]





texture = cv2.imread('/home/dcor/niskhizov/AdversarialRendering/mitsuba/colorful_pattern.jpg')#gsoup.generate_voronoi_diagram(512, 512, 1000)*0+255
proj_texture = gsoup.to_float(texture)

scene_dict = generate_random_scene()

scene = mi.load_dict(scene_dict)


params = mi.traverse(scene)

proj_tex_grad = torch.from_numpy(proj_texture).requires_grad_(True).cuda()  # Convert to (C, H, W) format for PyTorch

final_image = render(params, proj_tex_grad)

plt.imshow(final_image.detach().cpu().numpy(), aspect='auto')
import torchvision

resizer = torchvision.transforms.Resize((256, 256))
num_anchors = 10
learnable_anchors = torch.rand(num_anchors, 3, 10, 10).cuda()  # Example learnable anchors
learnable_anchors = learnable_anchors.requires_grad_(True)

proj_tex = torch.from_numpy(proj_texture).cuda() 
proj_tex_r = resizer(proj_tex.permute(2, 0, 1))
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
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=3, padding=1),  # 4x60x80 -> 8x60x80
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # 8x60x80 -> 16x60x80
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 8x60x80 -> 16x60x80
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 8x60x80 -> 16x60x80
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1), # 16x60x80 -> 8x60x80
            nn.LeakyReLU(),
            nn.Conv2d(8, 3, kernel_size=3, padding=1) )  # 8x60x80 -> 4x60x80
        


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        

learnable_anchors = torch.load('learnable_anchors.pth')
light_cnn_sd = torch.load('light_cnn.pth')
cnn_downsample_sd = torch.load('cnn_downsample.pth')
from torch import optim

cnn_ds = AnchorsCnnDownsample(3, 64).cuda()
model = LightCNN().cuda()    

cnn_ds.load_state_dict(cnn_downsample_sd)
model.load_state_dict(light_cnn_sd)

optimizer = optim.Adam([learnable_anchors,*cnn_ds.parameters(), *model.parameters()], lr=1e-3)

projected_anchors = []
for anchor in learnable_anchors:
    resized_anchor = resizer(anchor)  # Change to HWC format
    projected_anchors.append(render(params, resized_anchor.permute(1,2,0)))

mi.set_log_level(mi.LogLevel.Error)  # Suppress Mitsuba logs
import tqdm
proj_tex_r.shape
from torchmetrics.image import TotalVariation
tv = TotalVariation().cuda()
caltec_photos = '/home/dcor/niskhizov/caltech-101/101_ObjectCategories/'
ls = glob.glob(caltec_photos + '/*/*.jpg')

for epoch in tqdm.tqdm(range(10000)):

    random_caltech_photo = np.random.choice(ls)
    texture = cv2.imread(random_caltech_photo)#gsoup.generate_voronoi_diagram(512, 512, 1000)*0+255
    proj_texture = gsoup.to_float(texture)
    proj_tex = torch.from_numpy(proj_texture).cuda() 
    proj_tex_r = resizer(proj_tex.permute(2, 0, 1))

    scene_dict = generate_random_scene()
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)

    optimizer.zero_grad()

    # Resize the learnable anchors to match the input size
    projected_anchors = []
    true_proj = render(params, proj_tex_r.permute(1,2,0))

    for anchor in learnable_anchors:
        resized_anchor = resizer(anchor)  # Change to HWC format
        projected_anchors.append(render(params, resized_anchor.permute(1,2,0)))
    
    projected_anchors = torch.stack(projected_anchors) 
    # raise
    condition = cnn_ds(projected_anchors.permute(0,3,1,2))

    model_input = torch.cat([proj_tex_r, condition])

    
    output = model(model_input)
    
    tv_loss = tv(learnable_anchors) / 1000
    
    loss = torch.nn.functional.mse_loss(output, true_proj.permute(2,0,1))  + tv_loss# Ensure true_proj is in (C, H, W) format
    
    loss.backward()
    optimizer.step()
    if (epoch) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        plt.imshow(output.permute(1,2,0).detach().cpu().numpy(), aspect='auto')
        plt.show()
        plt.imshow(true_proj.detach().cpu().numpy(), aspect='auto')
        plt.show()

        # Save the model state
        torch.save(learnable_anchors, 'learnable_anchors.pth')
        torch.save(model.state_dict(), 'light_cnn.pth')
        torch.save(cnn_ds.state_dict(), 'cnn_downsample.pth')
        # Save the current learnable anchors
        
                