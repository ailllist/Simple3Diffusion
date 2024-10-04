
# Simple3Diffusion

This code is a **simplified version of Diffusion Probabilistic Models for 3D Point Cloud Generation (**https://github.com/luost26/diffusion-point-cloud**)**.

The purpose of this implementation is to help understand the basic flow of 3D Diffusion. Most of the code structure is identical to the original `diffusion-point-cloud` implementation. 
I have removed the parts related to Latent Shape from the original code and retained only the essential structure required for Diffusion to function.
Additionally, since the main objective is to comprehend the Diffusion flow, the code is designed to work for a single category only.
### Objectives of this code:

1. Understand the overall code flow of the Diffusion process.
2. Visually verify that Diffusion works correctly.

Since the basic Diffusion structure for 3D shapes has performance limitations, I have also removed all performance-related code and added a few simple visualization functionalities to complete this implementation.

### How to Run

Without any modifications, as long as the `shapenet.hdf5` file is located at `data/shapenet.hdf5`
with the appropriate libraries and dataset provided by the `diffusion-point-cloud` repository, 
the code can be run directly using `main.py`.

### Configurable Parameters

The adjustable parameters are in `main.py` lines 39-42:

```python
path = './data/shapenet.hdf5'
cate = "airplane"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw = False
```

You can modify these parameters to specify the category and dataset path.

### Important Note

This implementation is only for understanding the core code flow of the 3D Diffusion process.

### Results
(10000 iteration, 100 steps)
![Airplane](gifs/res_plane.gif)
![HeadPhone](gifs/res_headphone.gif)