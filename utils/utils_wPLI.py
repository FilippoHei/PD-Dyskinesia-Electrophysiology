"""
Functions for computing debiased weighted phase lag index (wPLI) and phase slope index (PSI).
"""
import os
import numpy as np
import pandas as pd
import sys
import math
from mne_connectivity import spectral_connectivity_epochs, phase_slope_index
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

sys.path.insert(0, './utils/')
from lib_data import DATA_IO
import utils_io

# Surrogate iteration function
def surrogate_iteration(seed, data, sfreq, fmin, fmax):
    '''
    Perform surrogate analysis by shuffling the data. 
    Shuffling is done across  the second channel (ECoG) and all trials (i.e., axis=0).
    Shuffled epochs resemble the null hypothesis of no connectivity.
    '''
    rng = np.random.default_rng(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled[:, 1, :], axis=0)

    conn_surr = spectral_connectivity_epochs(
        shuffled,
        method="wpli2_debiased",
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin, fmax=fmax,
        faverage=True,
        mt_adaptive=True,
        mt_low_bias=True,
        verbose=False,
    )
    return conn_surr.get_data()[2,:]

# Main function to compute wPLI and PSI with surrogates
def compute_wpli_psi(data, sfreq=2048, n_perm=1000, n_jobs=-1):
    """
    Compute debiased wPLI and PSI for LFP–ECoG pairs across trials.
    lfp_trials, ecog_trials: arrays (n_trials, n_times)
    """

    # Calculate and average within the frequency bands
    fmin=[4, 8, 12, 20, 60, 80],    # theta, alpha, beta_low, beta_high, gamma, gamma_III
    fmax=[8, 12, 20, 35, 90, 90],   # alpha ends at 12, gamma ends at 90

    conn_PLI = spectral_connectivity_epochs(
        data,
        method=['wpli2_debiased'],
        mode='multitaper',
        sfreq=sfreq,
        fmin=fmin, fmax=fmax,
        faverage=True,   # average inside each band
        mt_adaptive=True,
        mt_low_bias=True,
        verbose=False,
    )
    #.get_data()[:, 2]
    
    freqs = conn_PLI.freqs
    PLI_observed = conn_PLI.get_data()[2,:]  # (n_freqs,)

    # Surrogate iterations (parallelized)
    with tqdm_joblib(desc="Running surrogate iterations", total=n_perm) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(surrogate_iteration)(seed, data, sfreq, fmin, fmax)
            for seed in range(n_perm)
        )

    surrogates = np.vstack(results)

    # --- 3. Empirical p-values ---
    # two-sided test
    p_values = (np.sum(np.abs(surrogates) >= np.abs(PLI_observed), axis=0) + 1) / (n_perm + 1)
    
    # Phase slope index computed separately
    conn_PSI = phase_slope_index(
        data,
        sfreq=sfreq,
        mode='multitaper',
        fmin=fmin, fmax=fmax,
        mt_adaptive=True,
        mt_low_bias=True,
        verbose=False,
    )

    results = dict(freqs=np.vstack([fmin,fmax]), wpli=conn_PLI.get_data()[2,:], p_values=p_values, psi=conn_PSI.get_data()[2,:])
    return results


def get_wPLI_psi(dataset_LFP, dataset_ECOG, filename, condition, fs=2048, n_perm=1000, forceComputation=False):
    '''
    Compute debiased weighted phase lag index (wPLI) and phase slope index (PSI) between LFP and ECoG channel pairs across trials.

    This function matches LFP and ECoG events for each patient and channel, filters out epochs shorter than 512 samples,
    trims all remaining epochs to a common length, and computes connectivity measures using surrogate statistics.

    Args:
        dataset_LFP (pd.DataFrame): DataFrame containing LFP event recordings and metadata.
        dataset_ECOG (pd.DataFrame): DataFrame containing ECoG event recordings and metadata.
        filename (str): Filename to save the results DataFrame. Will be saved as a pickle file in ../events/coherence/.
        condition (str): Column name specifying which event recordings to use. Must be one of 'pre_event_recording', 'event_recording', or 'post_event_recording'.
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        n_perm (int, optional): Number of surrogate iterations for empirical p-value estimation. Default is 1000.
        forceComputation (bool, optional): If True, forces computation even if the output file already exists. Default is False.

    Returns:
        Results: Dataframe containing patient/channel info, frequency bands, wPLI, p-values, and PSI for each LFP-ECoG pair.
    '''
    
    # If filename has an any folder paths or extensions, remove them
    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0]
    
    # If the file already exists, skip computation
    if os.path.exists(DATA_IO.path_events + f"coherence/{filename}.pkl"): 
        if forceComputation:
            print("File already exists. But running computation anyway due to forceComputation=True.")
            print("File will be saved with an incremented filename if it already exists.")
        else:
            print("File already exists. Skipping computation.")
            return        
    
    Results = []
    for (patient, lfp_ch), group_lfp in dataset_LFP.groupby(["patient", "LFP_channel"]):

        # loop through each ECoG channel for this patient
        group_ecog_patient = dataset_ECOG[dataset_ECOG["patient"] == patient]
        for ecog_ch, group_ecog in group_ecog_patient.groupby("ECoG_channel"):
            print(f"Computing wPLI and PSI during events for patient {patient}, LFP {lfp_ch}, ECoG {ecog_ch}")
            # stack all tapping events for this patient + LFP channel
            # Make sure to only use events with the same laterality (i.e., ipsilateral) that occur in both event lists
            group_lfp = group_lfp[group_lfp["event_no"].isin(group_ecog["event_no"])]
            group_ecog = group_ecog[group_ecog["event_no"].isin(group_lfp["event_no"])]
            
            # Skip if either group is empty after filtering
            if group_lfp.empty or group_ecog.empty:
                print(f"Skipping patient {patient}, LFP {lfp_ch}, ECoG {ecog_ch}: no matching events.")
                continue
            
            
            # Only keep epochs where both arrays are >= 512 samples to ensure stable wPLI estimates also in the theta band range.
            arrays_lfp = [np.array(a) for a in group_lfp[condition]]
            arrays_lfp = [a for a in arrays_lfp if len(a) >= 512]
            
            arrays_ecog = [np.array(a) for a in group_ecog[condition]]
            arrays_ecog = [a for a in arrays_ecog if len(a) >= 512]
            
            if len(arrays_ecog) == 0 or len(arrays_lfp) == 0:
                print("No epochs with sufficient length.")
                continue
            
            # If ≥90% of trials reach a certain length → use that as target_len
            lengths = np.array([len(a) for a in arrays_lfp + arrays_ecog])
            target_len = math.floor(np.percentile(lengths, 10))  # 10th percentile keeps 90% of trials

            # Discard too-short trials and trim all remaining to target_len
            arrays_lfp, arrays_ecog = zip(*[(lfp, ecog) for lfp, ecog in zip(arrays_lfp, arrays_ecog) if len(lfp) >= target_len and len(ecog) >= target_len])
            arrays_lfp = [a[:target_len] for a in arrays_lfp]
            arrays_ecog = [a[:target_len] for a in arrays_ecog]

            # ensure same number of epochs
            assert len(arrays_lfp) == len(arrays_ecog), f"Number of epochs do not match for patient {patient}, LFP {lfp_ch}, ECoG {ecog_ch}"

            if len(arrays_lfp) != len(lengths)//2 or target_len != max(lengths):
                print(f"Using {len(arrays_lfp)} of {len(lengths)//2} tapping events with {target_len} samples ({int(1000*target_len/fs)} ms) out of originally {max(lengths)} samples for wPLI computation.")
            else:
                print(f"Using {len(arrays_lfp)} tapping events with {target_len} samples ({int(1000*target_len/fs)} ms) for wPLI computation.")

            # build data array (n_epochs, n_signals=2, n_times)
            data = np.stack([arrays_lfp, arrays_ecog], axis=1)
            
            # compute wPLI and PSI
            out = compute_wpli_psi(data, sfreq=fs, n_perm=n_perm)

            Results.append(dict(
                patient=patient,
                LFP_channel=lfp_ch,
                ECoG_channel=ecog_ch,
                freqs=out['freqs'],
                wpli=out['wpli'],
                p_values=out['p_values'],
                psi=out['psi'],
            ))

    utils_io.save_wPLI_df(pd.DataFrame(Results), filename)
    
    return pd.DataFrame(Results)

