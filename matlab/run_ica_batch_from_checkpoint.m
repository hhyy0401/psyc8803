function run_ica_batch_from_checkpoint(baseDir)
% RUN_ICA_BATCH_FROM_CHECKPOINT - Resume ICA cleaning where last stopped

if nargin < 1
    baseDir = '/Volumes/EEG Backup/data';
end

% === Add EEGLAB to path and initialize ===
eeglab_path = '/Volumes/EEG Backup/matlab/eeglab2024.2';
if exist(eeglab_path, 'dir')
    addpath(genpath(eeglab_path));
    eeglab; close; % Start EEGLAB and immediately close the GUI
else
    error('EEGLAB path not found at %s', eeglab_path);
end

input_dir  = fullfile(baseDir, 'preprocess', 'interval');
output_dir = fullfile(baseDir, 'clean');
log_path   = fullfile(output_dir, 'ica_manual_log.csv');
statusPath = fullfile(baseDir, 'status_log.mat');

if ~isfile(statusPath)
    error('Missing status_log.mat. Run scan_subjects_status.m first.');
end
load(statusPath, 'statusTable');

% Find next subject to process
nextIdx = find(statusTable.Preprocessed & ~statusTable.ICA_Cleaned, 1);
if isempty(nextIdx)
    disp('✅ All subjects have been ICA cleaned.');
    return;
end

subj_id = statusTable.Subject{nextIdx};
fprintf('\n🔁 Running ICA on Subject: %s\n\n', subj_id);

subj_input_path  = fullfile(input_dir, subj_id);
subj_output_path = fullfile(output_dir, subj_id);
if ~exist(subj_output_path, 'dir'), mkdir(subj_output_path); end

files = dir(fullfile(subj_input_path, '*_interval.set'));
if isempty(files)
    warning('⚠️ No interval .set files for %s. Skipping.', subj_id);
    return;
end

log_data = {};

for file = files'
    EEG = pop_loadset('filename', file.name, 'filepath', subj_input_path);
    EEG = eeg_checkset(EEG);

    % Trim first 500 ms
    EEG = pop_select(EEG, 'time', [0.5 EEG.xmax]);

    % Save original channel structure
    original_chanlocs = EEG.chanlocs;

    % Identify bad channels
    EEG_temp = clean_artifacts(EEG, ...
        'ChannelCriterion', 0.85, ...
        'FlatlineCriterion', 'off', ...
        'LineNoiseCriterion', 'off', ...
        'Highpass', 'off', ...
        'BurstCriterion', 'off', ...
        'WindowCriterion', 'off', ...
        'RepairChannels', false);

    removed_chans = setdiff({EEG.chanlocs.labels}, {EEG_temp.chanlocs.labels});
    if length(removed_chans) > 12
        removed_chans = removed_chans(1:12);
    end
    EEG = pop_select(EEG, 'nochannel', find(ismember({EEG.chanlocs.labels}, removed_chans)));
    EEG = eeg_checkset(EEG);

    % ICA
    EEG = pop_runica(EEG, 'extended', 1, 'interrupt', 'on');
    EEG.setname = [EEG.setname, '_ica'];
    EEG = eeg_checkset(EEG);
    
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0); % Ensure stored in EEGLAB memory
    EEG = iclabel(EEG); % ICLabel works on ALLEEG sync too
    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET); % Resync


    % Manual Component Rejection
    nIC = size(EEG.icaweights, 1);
    if nIC <= 35
        pop_selectcomps(EEG, 1:nIC);
    else
        pop_selectcomps(EEG, 1:35);
        pop_selectcomps(EEG, 36:nIC);
    end
    h = msgbox(sprintf('Reject ICs for %s\nClick OK to continue.', file.name));
    while ishandle(h), pause(0.5); end

    % Apply rejection
    EEG = eeg_checkset(EEG);
    comps_to_remove = find(EEG.reject.gcompreject);
    if ~isempty(comps_to_remove)
        EEG = pop_subcomp(EEG, comps_to_remove, 0);
    end

    % Interpolate
    if ~isempty(removed_chans)
        EEG = eeg_interp(EEG, original_chanlocs, 'spherical');
    end

    % Save cleaned file
    [~, name, ~] = fileparts(file.name);
    output_name = [name '_ica.set'];
    EEG = pop_saveset(EEG, 'filename', output_name, 'filepath', subj_output_path);
    fprintf('✅ Saved: %s/%s\n', subj_output_path, output_name);

    % Log
    log_data = [log_data; {subj_id, file.name, length(removed_chans), strjoin(removed_chans, ',')}];
end

% Append to ICA log
if exist(log_path, 'file')
    old = readtable(log_path);
    log_table = [old; cell2table(log_data, ...
        'VariableNames', {'Subject', 'InputFile', 'NumBadChannels', 'RemovedChannels'})];
else
    log_table = cell2table(log_data, ...
        'VariableNames', {'Subject', 'InputFile', 'NumBadChannels', 'RemovedChannels'});
end
writetable(log_table, log_path);

% Update status log
statusTable.ICA_Cleaned(nextIdx) = true;
save(statusPath, 'statusTable');

fprintf('\n🎉 Done with subject %s\n', subj_id);
end
