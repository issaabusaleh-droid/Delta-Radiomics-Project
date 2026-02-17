import pandas as pd
import os

def extract_features_for_patient(patient_dir, timepoint):
    """
    Extract features from PET and mask at given timepoint.
    Returns a dictionary of feature names and values.
    """
    pet_path = os.path.join(patient_dir, timepoint, 'PET.nii.gz')
    mask_path = os.path.join(patient_dir, timepoint, 'mask.nii.gz')
    # CT can also be used, but for PET/CT we can extract from PET only.
    result = extractor.execute(pet_path, mask_path)
    # result is a collections.OrderedDict; convert to flat dict
    features = {}
    for key, val in result.items():
        if isinstance(val, (int, float)):
            features[f"{timepoint}_{key}"] = val
    return features

# Iterate over patients
all_features = []
patients = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d)) and d.startswith('patient')]
labels_df = pd.read_csv('data/labels.csv', index_col='patient_id')

for patient in patients:
    patient_dir = os.path.join('data', patient)
    try:
        feat_t0 = extract_features_for_patient(patient_dir, 'T0')
        feat_t1 = extract_features_for_patient(patient_dir, 'T1')
        combined = {**feat_t0, **feat_t1}
        combined['patient_id'] = patient
        combined['response'] = labels_df.loc[patient, 'response']
        all_features.append(combined)
    except Exception as e:
        print(f"Error with {patient}: {e}")

df_features = pd.DataFrame(all_features)
df_features.to_csv('all_features.csv', index=False)
