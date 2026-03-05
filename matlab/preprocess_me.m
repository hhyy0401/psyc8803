% ============================================
% PREPROC (No ICA), Makoto-aligned order
% *** NO SPLIT HERE (split will happen post-ICA in a separate script) ***
% NEVER drop E65/Cz (measured Cz is preserved)
% Order:
%   Resample -> HP 1 Hz
%   TEMP full-rank avg reref (de-zero E65)  <-- protective
%   Remove bad channels (channels-only)
%   Interpolate removed channels
%   FINAL full-rank avg reref (Makoto)
%   Remove bad sections (ASR/window, no channel removal)
%   (opt) LP 40 Hz
%   Save as EEGLAB .set (pop_saveset)
% Logs + metrics CSV included
% ============================================

[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% -------- CONFIG --------
raw_dir      = '/Volumes/T7/data/raw';
output_dir   = '/Volumes/T7/data/preprocess';

target_fs    = 250;      % resample target
hp_pre_Hz    = 1.0;      % high-pass before cleaning
lp_post_Hz   = 40;       % low-pass after cleaning (set [] to skip)

% Channel removal (FIRST PASS, channels only)
flatlineSec  = 5;        % flatline threshold in seconds ('off' to disable)
chanCorr     = 0.80;     % channel correlation threshold (0.80–0.90 typical)
lineNoiseSD  = 4;        % line-noise criterion in SDs

% Section removal (SECOND PASS, segments only)
burstCrit     = 40;      % ASR burst criterion (25–30 gentler)
winCrit       = 0.5;     % fraction of contaminated windows (0–1)
burstRejectOn = 'on';    % reject contaminated windows

% Labels to protect (NEVER remove)
protectLabels = {'E65','Cz'};

% -------- SUBSET CONTROL (process only a portion of the dataset) --------
subset_mode     = 'subjects';   % 'subjects' (default) or 'files'
subset_fraction = 0.5;          % take top 50%
start_from      = 'lowest';    % 'highest' (default) or 'lowest'

% -------- LOGS --------
if ~exist(output_dir,'dir'), mkdir(output_dir); end
logs_dir = fullfile(output_dir, 'clean_logs');
if ~exist(logs_dir,'dir'), mkdir(logs_dir); end

% Size/label CSV
log_file = fullfile(output_dir, 'EEGpreproc_log.csv');
fid = fopen(log_file, 'w');
if fid == -1, error('Could not open log file at %s', log_file); end
fprintf(fid, 'Subject,Recording,SizeMB,AssignedLabel\n');

% Metrics CSV
metrics_file = fullfile(output_dir, 'EEGclean_metrics.csv');
new_metrics  = ~exist(metrics_file, 'file');
metrics_fid  = fopen(metrics_file, 'a');
if metrics_fid == -1, error('Could not open metrics file: %s', metrics_file); end
if new_metrics
    fprintf(metrics_fid, ['Subject,Label,Cond,Srate,DurBeforeSec,DurAfterSec,KeptPct,',...
        'ChansBefore,ChansAfterInterp,ChansRemoved_ChannelPass,RemovedLabels_ChannelPass,',...
        'BoundariesAdded_Sections,ReinsertedCz,SavePath,Filename\n']);
end

% -------- FIND .mff (KEEP INDEXING ALIGNED) --------
mff_dirs = dir(fullfile(raw_dir, '*.mff'));
mff_dirs = mff_dirs([mff_dirs.isdir]);

recording_names = {mff_dirs.name};
tok = regexp(recording_names, '^(19\d{3})', 'tokens', 'once');

mask = ~cellfun(@isempty, tok);         % which entries actually matched the subject pattern
mff_dirs = mff_dirs(mask);              % shrink mff_dirs to match
tok      = tok(mask);

subject_ids     = cellfun(@(x) x{1}, tok, 'UniformOutput', false);
unique_subjects = unique(subject_ids);

% Plugins
hasFullRank = (exist('fullRankAveRef','file') == 2);

% -------- OPTIONAL: SUBSET SELECTION (start at highest index, take half) --------
switch lower(subset_mode)
    case 'subjects'
        subj_num = str2double(unique_subjects);  % e.g., "19031" -> 19031
        if strcmpi(start_from,'highest')
            [~, ord] = sort(subj_num, 'descend');
        else
            [~, ord] = sort(subj_num, 'ascend');
        end
        unique_subjects = unique_subjects(ord);

        keepN = max(1, floor(numel(unique_subjects) * subset_fraction));
        unique_subjects = unique_subjects(1:keepN);

        fprintf('[SUBSET] Processing %d/%d subjects (mode=subjects, start=%s, frac=%.2f). Range: %s → %s\n', ...
            keepN, numel(subj_num), start_from, subset_fraction, unique_subjects{1}, unique_subjects{end});

    case 'files'
        subj_num_each = str2double(subject_ids);
        if strcmpi(start_from,'highest')
            [~, ord] = sort(subj_num_each, 'descend');
        else
            [~, ord] = sort(subj_num_each, 'ascend');
        end

        mff_dirs    = mff_dirs(ord);
        subject_ids = subject_ids(ord);

        keepN       = max(1, floor(numel(mff_dirs) * subset_fraction));
        mff_dirs    = mff_dirs(1:keepN);
        subject_ids = subject_ids(1:keepN);

        unique_subjects = unique(subject_ids, 'stable');

        if ~isempty(unique_subjects)
            fprintf('[SUBSET] Processing top %d/%d files across %d subjects (mode=files, start=%s, frac=%.2f). First subj: %s\n', ...
                keepN, numel(ord), numel(unique_subjects), start_from, subset_fraction, unique_subjects{1});
        else
            warning('[SUBSET] No subjects left after subsetting!');
        end

    otherwise
        fprintf('[SUBSET] Disabled; processing all subjects.\n');
end

% -------- LOOP SUBJECTS --------
for s = 1:length(unique_subjects)
    subj_id    = unique_subjects{s};
    subj_idxs  = find(strcmp(subject_ids, subj_id));
    subj_files = mff_dirs(subj_idxs);

    fprintf('\nSubject %s — %d recordings\n', subj_id, length(subj_files));

    % Compute sizes
    sizes = zeros(1, length(subj_files));
    for i = 1:length(subj_files)
        try
            listing  = dir(fullfile(raw_dir, subj_files(i).name));
            contents = listing(~[listing.isdir]);
            sizes(i) = sum([contents.bytes]) / 1024^2;
        catch
            sizes(i) = 0;
        end
    end

    % Prefer pitch/duration by filename; fallback to largest two >100MB
    inferred = strings(1, length(subj_files));
    for i = 1:length(subj_files)
        inferred(i) = inferTaskFromName(subj_files(i).name); % 'pitch'|'duration'|'eyes'|'unknown'
    end
    is_pd   = ismember(lower(inferred), {'pitch','duration'});
    pd_idxs = find(is_pd);

    selected_files  = [];
    selected_labels = strings(0);
    selected_sizes  = [];

    if numel(pd_idxs) >= 2
        [~, ord] = sort(sizes(pd_idxs), 'descend');
        pick = pd_idxs(ord(1:2));
        selected_files  = mff_dirs(pick);
        selected_labels = inferred(pick);
        selected_sizes  = sizes(pick);
    else
        valid_idx = find(sizes > 100);
        if numel(valid_idx) < 2
            fprintf('Skipping %s — not enough recordings >100 MB\n', subj_id);
            continue;
        end
        [~, big_ord]   = sort(sizes(valid_idx), 'descend');
        pick           = valid_idx(big_ord(1:2));
        selected_files = mff_dirs(pick);
        selected_labels= inferred(pick);
        selected_sizes = sizes(pick);
    end

    % Write selection to size/label log
    for i = 1:numel(selected_files)
        fprintf(fid, '%s,%s,%.1f,%s\n', subj_id, selected_files(i).name, selected_sizes(i), char(selected_labels(i)));
    end

    % ------ LOAD & PREPROCESS WHOLE RECORDING (NO SPLIT) ------
    for i = 1:numel(selected_files)
        label      = selected_labels(i);
        mff_folder = fullfile(raw_dir, selected_files(i).name); % absolute path

        try
            setpref('eeglab','mff_import_mode','script');
            setpref('eeglab','mff_eventtypefield','code'); % use 'code' for NetStation
            EEG = pop_mffimport(mff_folder);
        catch ME
            warning('Failed to load %s (%s)', selected_files(i).name, ME.message);
            continue;
        end

        EEG.setname = sprintf('%s_%s_raw', subj_id, char(label));

        % sort events by latency for safety
        if isfield(EEG,'event') && ~isempty(EEG.event)
            [~, ord] = sort([EEG.event.latency]);
            EEG.event = EEG.event(ord);
        else
            warning('Skipping %s - %s: no events', subj_id, char(label));
            continue;
        end

        % --- Process the whole stream without splitting ---
        cond = "whole";          % single condition
        EEGc = EEG;              % working copy

        % Original montage for interpolation reference
        original_chanlocs = EEGc.chanlocs;

        % 1) Resample
        if EEGc.srate ~= target_fs
            EEGc = pop_resample(EEGc, target_fs);
        end

        % 2) High-pass
        EEGc = pop_eegfiltnew(EEGc, hp_pre_Hz, []);

        % ---- Cache measured Cz/E65 ASAP ----
        lbls0 = {EEGc.chanlocs.labels};
        cz_idx = find(strcmpi(lbls0,'E65') | strcmpi(lbls0,'Cz'), 1);
        if isempty(cz_idx)
            error('Cz/E65 not found in chanlocs for %s %s %s.', subj_id, char(label), char(cond));
        end
        cz_data_measured = EEGc.data(cz_idx,:);
        cz_loc_measured  = EEGc.chanlocs(cz_idx);

        % 3) TEMP full-rank avg reref (de-zero E65 so it survives channel cleaner)
        if hasFullRank
            EEGc = fullRankAveRef(EEGc);
        else
            EEGc = pop_reref(EEGc, [], 'refloc', struct('labels','initialRef'));
            EEGc = pop_select(EEGc, 'nochannel', {'initialRef'});
        end

        % 4) Remove BAD CHANNELS (channels-only)
        chans_before = numel({EEGc.chanlocs.labels});
        log_chan = fullfile(logs_dir, sprintf('%s_%s_%s_channels.txt', subj_id, char(label), char(cond)));
        clog1 = '';
        try
            if ~isempty(which('clean_artifacts'))
                [clog1, EEGc] = evalc( ...
                    ['clean_artifacts(EEGc, ''FlatlineCriterion'', ',num2str(flatlineSec), ...
                     ', ''Highpass'', ''off'', ''ChannelCriterion'', ',num2str(chanCorr), ...
                     ', ''LineNoiseCriterion'', ',num2str(lineNoiseSD), ...
                     ', ''BurstCriterion'', ''off'', ''WindowCriterion'', ''off'', ', ...
                     '''BurstRejection'', ''off'', ''Distance'', ''Euclidean'')'] );
            elseif ~isempty(which('pop_clean_rawdata'))
                [clog1, EEGc] = evalc( ...
                    ['pop_clean_rawdata(EEGc, ''FlatlineCriterion'', ',num2str(flatlineSec), ...
                     ', ''ChannelCriterion'', ',num2str(chanCorr), ...
                     ', ''LineNoiseCriterion'', ',num2str(lineNoiseSD), ...
                     ', ''Highpass'', ''off'', ''BurstCriterion'', ''off'', ', ...
                     '''WindowCriterion'', ''off'', ''BurstRejection'', ''off'')'] );
            else
                [clog1, EEGc] = evalc('clean_rawdata(EEGc, flatlineSec, chanCorr, lineNoiseSD, -1, -1, -1)');
            end
        catch MEc
            fidt=fopen(log_chan,'w'); if fidt~=-1, fprintf(fidt,'%s\nERROR: %s\n',clog1,MEc.message); fclose(fidt); end
            rethrow(MEc);
        end
        fidt=fopen(log_chan,'w'); if fidt~=-1, fprintf(fidt,'%s',clog1); fclose(fidt); end

        % If Cz was removed, reinsert measured Cz/E65 (NEVER interpolate Cz)
        lbls_after_chan = {EEGc.chanlocs.labels};
        reinsertedCz = 0;
        if ~any(strcmpi(lbls_after_chan,'E65') | strcmpi(lbls_after_chan,'Cz'))
            EEGc.nbchan          = EEGc.nbchan + 1;
            EEGc.data(end+1,:)   = cz_data_measured;
            EEGc.chanlocs(end+1) = cz_loc_measured;
            reinsertedCz = 1;
            fprintf('Reinserted measured Cz/E65 after channel cleaning for %s-%s-%s.\n', subj_id, char(label), char(cond));
        end

        % Determine channels removed by channel pass (relative to pre-clean labels)
        removed_labels = setdiff(lbls0, {EEGc.chanlocs.labels}, 'stable');
        removed_labels = removed_labels(~ismember(lower(removed_labels), lower(protectLabels)));
        chans_removed  = numel(removed_labels);

        % 5) Interpolate removed channels (back to full montage)
        EEGc = pop_interp(EEGc, original_chanlocs, 'spherical');
        chans_after_interp = numel({EEGc.chanlocs.labels});

        % 6) FINAL full-rank avg reref (Makoto)
        if hasFullRank
            EEGc = fullRankAveRef(EEGc);
        else
            EEGc = pop_reref(EEGc, [], 'refloc', struct('labels','initialRef'));
            EEGc = pop_select(EEGc, 'nochannel', {'initialRef'});
        end

        % 7) Remove BAD SECTIONS (segments-only; no channel removal)
        dur_before_sec = EEGc.pnts / EEGc.srate;
        b0 = sum(strcmpi({EEGc.event.type}, 'boundary'));
        log_sec = fullfile(logs_dir, sprintf('%s_%s_%s_sections.txt', subj_id, char(label), char(cond)));
        clog2 = '';
        try
            if ~isempty(which('clean_artifacts'))
                [clog2, EEGc] = evalc( ...
                    ['clean_artifacts(EEGc, ''FlatlineCriterion'', ''off'', ''Highpass'', ''off'', ', ...
                     '''ChannelCriterion'', ''off'', ''LineNoiseCriterion'', ',num2str(lineNoiseSD), ...
                     ', ''BurstCriterion'', ',num2str(burstCrit), ...
                     ', ''WindowCriterion'', ',num2str(winCrit), ...
                     ', ''BurstRejection'', ''', burstRejectOn, ''', ''Distance'', ''Euclidean'')'] );
            elseif ~isempty(which('pop_clean_rawdata'))
                [clog2, EEGc] = evalc( ...
                    ['pop_clean_rawdata(EEGc, ''FlatlineCriterion'', ''off'', ''ChannelCriterion'', ''off'', ', ...
                     '''LineNoiseCriterion'', ',num2str(lineNoiseSD), ...
                     ', ''Highpass'', ''off'', ''BurstCriterion'', ',num2str(burstCrit), ...
                     ', ''WindowCriterion'', ',num2str(winCrit), ...
                     ', ''BurstRejection'', ''', burstRejectOn, ''')'] );
            else
                [clog2, EEGc] = evalc('clean_rawdata(EEGc, -1, -1, -1, -1, burstCrit, winCrit)');
            end
        catch MEc
            fidt=fopen(log_sec,'w'); if fidt~=-1, fprintf(fidt,'%s\nERROR: %s\n',clog2,MEc.message); fclose(fidt); end
            rethrow(MEc);
        end
        fidt=fopen(log_sec,'w'); if fidt~=-1, fprintf(fidt,'%s',clog2); fclose(fidt); end
        b1 = sum(strcmpi({EEGc.event.type}, 'boundary'));
        boundaries_added = max(b1 - b0, 0);
        dur_after_sec    = EEGc.pnts / EEGc.srate;
        kept_pct         = 100 * dur_after_sec / max(dur_before_sec, eps);

        % 8) Low-pass (optional)
        if ~isempty(lp_post_Hz)
            EEGc = pop_eegfiltnew(EEGc, [], lp_post_Hz);
        end

        % 9) Save as EEGLAB .set (single "whole" dataset)
        save_path  = fullfile(output_dir, 'Whole', subj_id);
        if ~exist(save_path,'dir'), mkdir(save_path); end
        basefile   = sprintf('%s_%s_%s', subj_id, char(label), char(cond));
        filename   = sprintf('%s.set', basefile);

        EEGc.setname = sprintf('%s_%s_%s_preproc', subj_id, char(label), char(cond));
        EEGc = eeg_checkset(EEGc);
        pop_saveset(EEGc, 'filename', filename, 'filepath', save_path, 'version', '7.3');

        fprintf('✔ %s — %s %s preprocessed (WHOLE) & saved to %s\n', ...
            subj_id, char(label), char(cond), fullfile(save_path, filename));

        % Metrics row
        fprintf(metrics_fid, '%s,%s,%s,%.3f,%.2f,%.2f,%.1f,%d,%d,%d,"%s",%d,%d,"%s","%s"\n', ...
            subj_id, char(label), char(cond), EEGc.srate, ...
            dur_before_sec, dur_after_sec, kept_pct, ...
            chans_before, chans_after_interp, chans_removed, ...
            strjoin(removed_labels,';'), boundaries_added, reinsertedCz, save_path, filename);
    end
end

% -------- CLOSE LOGS --------
fclose(fid);
fclose(metrics_fid);
fprintf('\nDone!\n- Size/label log: %s\n- Metrics CSV:     %s\n- Cleaner logs:    %s\n', log_file, metrics_file, logs_dir);

% === Helpers ===
function out = capitalizeFirst(str)
    if isempty(str), out = str; return; end
    out = [upper(str(1)) lower(str(2:end))];
end

function task = inferTaskFromName(nameStr)
    % Robust, case-insensitive inference of task label from filename
    n = lower(string(nameStr));
    if contains(n, "pitch") || contains(n, "_p") || contains(n, "p_")
        task = "pitch";
    elseif contains(n, "duration") || contains(n, "_d") || contains(n, "d_")
        task = "duration";
    elseif contains(n, "eyes") || contains(n, "rest") || contains(n, "eo") || contains(n, "ec")
        task = "eyes";
    else
        task = "unknown";
    end
end
