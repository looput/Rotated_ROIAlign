import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2

import torchvision.utils as vutils
from roi_align_rotate import ROIAlignRotated

pooler_rotated=ROIAlignRotated((32,192), spatial_scale = (1.), sampling_ratio = 0)

image=cv2.imread('IMG_0451.jpg')

with open('IMG_0451.gt') as f:
    lines=f.readlines()
    rectes=[]

for line in lines:
    line=line.split(' ',7)[2:]
    rectes.append([float(num.strip('\n')) for num in line])


device=torch.device('cuda')
rectes=torch.from_numpy(np.array(rectes)).to(device=device,dtype=torch.float32)    

rectes[:,:2]=rectes[:,:2]+rectes[:,2:4]/2
rectes[:,-1]=-1*rectes[:,-1]*180/np.pi

image_tensor=torch.from_numpy(image[np.newaxis,:,:,:]).to(device=device)

ids=torch.full((rectes.shape[0], 1), 0, dtype=torch.float32, device=device)
rois=torch.cat([ids,rectes],dim=1)

image_tensor=image_tensor.transpose(1,3).transpose(2,3).to(torch.float32)
image_roi_bbox=pooler_rotated(image_tensor,rois)

image_show=vutils.make_grid(image_roi_bbox[:,...], normalize=True, scale_each=True,nrow=2)
image_show=image_show.detach().permute(1,2,0).cpu().numpy()

plt.imshow(image_show)
plt.show()