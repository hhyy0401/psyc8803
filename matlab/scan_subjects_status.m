function statusTable = scan_subjects_status(baseDir)
% SCAN_SUBJECTS_STATUS - Scans EEG directory structure for processing progress.
% Usage: statusTable = scan_subjects_status('/Volumes/EEG Backup')

if nargin < 1
    baseDir = '/Volumes/EEG Backup/data';
end

% Get list of subject folders in preprocess/interval
intervalDir = fullfile(baseDir, 'preprocess', 'interval');
subjectDirs = dir(intervalDir);
subjectDirs = subjectDirs([subjectDirs.isdir] & ~startsWith({subjectDirs.name}, '.'));
subjectIDs = {subjectDirs.name};

% Initialize status matrix
status = [];

for i = 1:length(subjectIDs)
    subj = subjectIDs{i};

    % Define expected file paths
    preproc_pitch = fullfile(baseDir, 'preprocess', 'interval', subj, [subj '_pitch_interval.set']);
    preproc_dura  = fullfile(baseDir, 'preprocess', 'interval', subj, [subj '_duration_interval.set']);

    clean_pitch = fullfile(baseDir, 'clean', subj, [subj '_pitch_interval_ica.set']);
    clean_dura  = fullfile(baseDir, 'clean', subj, [subj '_duration_interval_ica.set']);

    epoched_dir = fullfile(baseDir, 'epoched', subj);
    extracted_file = fullfile(baseDir, 'output', 'metrics', [subj '_metrics.csv']);

    % Check existence
    isPreprocessed = isfile(preproc_pitch) && isfile(preproc_dura);
    isCleaned      = isfile(clean_pitch) && isfile(clean_dura);
    isEpoched      = isfolder(epoched_dir) && ~isempty(dir(fullfile(epoched_dir, '*.set')));
    isExtracted    = isfile(extracted_file);

    % Append to status
    status = [status; {subj, isPreprocessed, isCleaned, isEpoched, isExtracted}];
end

% Convert to table
statusTable = cell2table(status, ...
    'VariableNames', {'Subject', 'Preprocessed', 'ICA_Cleaned', 'Epoched', 'Extracted'});

% Save log for dashboard resume
save(fullfile(baseDir, 'status_log.mat'), 'statusTable');

% Display preview
disp(statusTable);
end
