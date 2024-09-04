import torch
import numpy as np
import nibabel as nib
import data_utils as du

# Load three tensors
tensor1 = torch.load('/EVENetCNN/pred_prob_e3.pt')
tensor2 = torch.load('/EVENetCNN/pred_prob_md.pt')
tensor3 = torch.load('/EVENetCNN/pred_prob_fa.pt')

# Calculate the average probability
average_tensor = (tensor1 + tensor2 + tensor3) / 3

average_tensor = average_tensor.float()

# Calculate cross-entropy as uncertainty
probs = torch.softmax(average_tensor, dim=-1)
log_probs = torch.log(probs)
uncertainty = -1.0 * (probs * log_probs).sum(dim=-1)

# Save uncertainty as a heatmap
uncertainty_nii = nib.Nifti1Image(uncertainty.numpy(), np.eye(4))
nib.save(uncertainty_nii, '/EVENetCNN/deep_ensemble_heatmap.nii.gz')

# Calculate the class with the highest probability for each voxel
_, max_prob_class = torch.max(average_tensor, dim=-1)

# Save the classification results
# max_prob_class_nii = nib.Nifti1Image(max_prob_class.numpy(), np.eye(4))
# nib.save(max_prob_class_nii, '/EVENetCNN/deep_ensemble_pred.nii.gz')

# Map to FreeSurfer label space
muLut = du.read_classes_from_lut("/EVENetCNN/config/EVENet_ColorLUT.tsv")
myLabels = muLut["ID"].values

pred_classes = du.map_label2aparc_aseg(max_prob_class, myLabels)
pred_classes = du.split_cortex_labels(pred_classes.cpu().numpy())

np_data = pred_classes if isinstance(pred_classes, np.ndarray) else pred_classes.cpu().numpy()

orig = nib.load("/PATH/TO/SUBJECT_DATA/SUB_X/fa.nii.gz")  # Update this path with actual subject data
header = orig.header
save_as = "/EVENetCNN/ensemble_pred"

du.save_image(header, orig.affine, np_data, save_as, dtype=np.int16)
