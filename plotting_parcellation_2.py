import nibabel as nib
from nilearn import datasets, image, plotting

# Load the Desikan-Killiany atlas
atlas_dataset = datasets.fetch_atlas_aal()
atlas_filename = atlas_dataset.maps

# Load a template brain image (e.g., MNI152 template)
template = datasets.load_mni152_template()

# Apply the atlas parcellation to the template brain image
parcellation_img = image.
parcellations.img_to_roi_labels(template, roi_values=np.arange(1, 69), mask_img=atlas_filename)

# Save the parcellated image
#parcellation_output = 'path_to_save_parcellated_template.nii.gz'
#nib.save(parcellation_img, parcellation_output)

# Display the parcellated image (optional)
plotting.plot_roi(parcellation_img, cmap='Paired')
plotting.show()