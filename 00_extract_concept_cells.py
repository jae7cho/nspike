import os
import glob
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.stats import f_oneway, ttest_ind
import warnings

warnings.filterwarnings('ignore')

# [Keep the check_concept_cell function exactly the same here]
def check_concept_cell(unit_spike_times, trials_df):
    delay = 0.2
    offset = 1.0
    rates, image_ids = [], []
    valid_trials = trials_df.dropna(subset=['timestamps_Encoding1', 'loadsEnc1_PicIDs'])
    for _, trial in valid_trials.iterrows():
        t_start = trial['timestamps_Encoding1'] + delay
        t_end = trial['timestamps_Encoding1'] + offset
        spikes = np.sum((unit_spike_times >= t_start) & (unit_spike_times < t_end))
        rates.append(spikes / (offset - delay))
        image_ids.append(trial['loadsEnc1_PicIDs'])
    rates, image_ids = np.array(rates), np.array(image_ids)
    unique_ids = np.unique(image_ids)
    groups = [rates[image_ids == img_id] for img_id in unique_ids]
    if len(groups) < 2: return False, {}
    f_stat, p_anova = f_oneway(*groups)
    if p_anova < 0.05:
        mean_rates = [np.mean(g) for g in groups]
        pref_id = unique_ids[np.argmax(mean_rates)]
        pref_rates = rates[image_ids == pref_id]
        non_pref_rates = rates[image_ids != pref_id]
        if len(pref_rates) > 0 and len(non_pref_rates) > 0:
            t_stat, p_ttest = ttest_ind(pref_rates, non_pref_rates, equal_var=False)
            if p_ttest < 0.05 and np.mean(pref_rates) > np.mean(non_pref_rates):
                return True, {
                    'preferred_image_id': pref_id,
                    'anova_p': p_anova,
                    'ttest_p': p_ttest,
                    'pref_hz': np.mean(pref_rates),
                    'base_hz': np.mean(non_pref_rates)
                }
    return False, {}

# Updated cache_dataset function
def cache_dataset(base_dir, custom_save_dir):
    """Scans the dataset and saves CSVs to a custom directory."""
    
    # Ensure the custom save directory exists
    os.makedirs(custom_save_dir, exist_ok=True)
    
    search_pattern = os.path.join(base_dir, "**", "*ses-2*ecephys+image.nwb")
    nwb_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(nwb_files)} NWB files to process.\n")
    
    for filepath in nwb_files:
        filename = os.path.basename(filepath)
        print(f"Processing: {filename}")
        
        # --- THE NEW SAVE PATH LOGIC ---
        # Create a new filename for the CSV based on the original NWB name
        csv_filename = filename.replace(".nwb", "_concept_cells.csv")
        # Save it into the custom directory instead of the NWB's directory
        save_path = os.path.join(custom_save_dir, csv_filename)
        
        if os.path.exists(save_path):
            print(f" -> Cache already exists at {save_path}! Skipping.\n")
            continue
            
        concept_cells = []
        
        with NWBHDF5IO(filepath, 'r') as io:
            nwb = io.read()
            trials_df = nwb.intervals['trials'].to_dataframe()
            units_df = nwb.units.to_dataframe()
            
            for unit_id in units_df.index:
                spike_times = nwb.units.get_unit_spike_times(unit_id)
                is_cc, stats = check_concept_cell(spike_times, trials_df)
                
                if is_cc:
                    stats['unit_id'] = unit_id
                    if 'electrodes' in units_df.columns:
                        try:
                            stats['location'] = units_df.loc[unit_id, 'electrodes']['location'].values[0]
                        except:
                            stats['location'] = 'unknown'
                    concept_cells.append(stats)
                    
        if concept_cells:
            cc_df = pd.DataFrame(concept_cells)
            cols = ['unit_id', 'location', 'preferred_image_id', 'anova_p', 'ttest_p', 'pref_hz', 'base_hz']
            cc_df = cc_df[[c for c in cols if c in cc_df.columns]]
            cc_df.to_csv(save_path, index=False)
            print(f" -> Saved {len(cc_df)} Concept Cells to {save_path}\n")
        else:
            print(f" -> No Concept Cells found in this session.\n")

if __name__ == "__main__":
    HUMAN_DATA_ROOT = "/Volumes/fetty/000469/"
    
    # DEFINE YOUR CUSTOM SAVE PATH HERE
    CUSTOM_CACHE_DIR = "/Users/cwook/Documents/humac/data/concept_cells/"
    
    cache_dataset(HUMAN_DATA_ROOT, CUSTOM_CACHE_DIR)