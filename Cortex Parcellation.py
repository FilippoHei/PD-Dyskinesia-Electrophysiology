import nibabel as nib
import nilearn.plotting as plotting
from nilearn.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the AAL3 NIfTI file
aal3_atlas_path = 'AAL3.nii'  # Replace with your file path
aal3_img        = nib.load(aal3_atlas_path)
aal3_data       = aal3_img.get_fdata()

ecog_contacts = np.array([
    [20.0, -56.0, 75.5],
    [24.5, -46.75, 78.25],
    [28.25, -36.75, 78.5],
    [32.0, -27.0, 76.5],
    [35.0, -17.25, 73.25]
])

print(f"Number of ECoG Contacts: {len(ecog_contacts)}")

# Step 3: Convert MNI Coordinates to Voxel Indices

affine = aal3_img.affine  # Get the affine matrix

def MNI_to_voxel(ECoG_MNI_coordinates, AAL3_affine_matrix):
    # Convert MNI coordinates to voxel indices using the affine matrix.
    return np.round(np.linalg.inv(AAL3_affine_matrix).dot(np.append(ECoG_MNI_coordinates, 1))[:3]).astype(int)

# Convert all ECoG contacts from MNI to voxel coordinates
voxel_indices = np.array([MNI_to_voxel(contact, affine) for contact in ecog_contacts])

# Step 4: Map Voxel Indices to AAL3 Atlas Labels

def get_AAL3_label(voxel_coord, aal_data):
    """Get the label from AAL3 atlas data for a given voxel coordinate."""
    x, y, z = voxel_coord
    # Ensure the voxel is within the atlas bounds
    if (0 <= x < aal_data.shape[0] and
        0 <= y < aal_data.shape[1] and
        0 <= z < aal_data.shape[2]):
        return aal_data[x, y, z]
    else:
        return 0  # Return 0 if voxel is outside the atlas

# Get AAL3 region labels for each ECoG contact
aal3_labels = [get_AAL3_label(voxel, aal3_data) for voxel in voxel_indices]


plt.show()
