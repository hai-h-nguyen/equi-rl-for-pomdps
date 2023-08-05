from escnn import gspaces                                          
from escnn import nn                                               
import torch              
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

image_width = 4
image_height = 4


grp_act = gspaces.rot2dOnR2(4)
feat_type_in  = nn.FieldType(grp_act, [grp_act.trivial_repr])

x = torch.randint(0, 2, (1, 1, image_height, image_width))

plt.subplot(231)
plt.imshow(x[0, 0, :].numpy())
x = feat_type_in(x)

print(x.tensor.squeeze())

plot_idices = [232, 233, 234, 235]
i = 0

for g in grp_act.testing_elements:
    print(x.transform(g).tensor.squeeze())
    plt.subplot(plot_idices[i])
    plt.imshow(x.transform(g).tensor[0, 0, :].numpy())
    i += 1
plt.show()
