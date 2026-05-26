function clean_data = preprocess_eeg_data(file_path, cfg)
  
    eeglab_options = {'gui', 'off'};
    [folder_path, file_name, ext] = fileparts(file_path);
    fprintf('Đang xử lý: %s\n', file_name);

    try
      
        EEG = pop_loadset('filename', [file_name, ext], 'filepath', folder_path);

        if EEG.srate ~= cfg.target_sfreq
            EEG = pop_resample(EEG, cfg.target_sfreq);
        end

        EEG = pop_eegfiltnew(EEG, 'locutoff', cfg.notch_freqs(1)-1, 'hicutoff', cfg.notch_freqs(1)+1, 'revfilt', 1);
        EEG = pop_eegfiltnew(EEG, 'locutoff', cfg.bandpass(1), 'hicutoff', cfg.bandpass(2));

        variances = var(EEG.data, 0, 2);
        Q1 = prctile(variances, 25);
        Q3 = prctile(variances, 75);
        IQR_val = Q3 - Q1;
        bad_idx = find(variances < (Q1 - 1.5 * IQR_val) | variances > (Q3 + 1.5 * IQR_val));
        
        if ~isempty(bad_idx)
            fprintf('  -> Nội suy %d kênh hỏng.\n', length(bad_idx));
            EEG = pop_interp(EEG, bad_idx, 'spherical');
        end

        actual_rank = min(cfg.ica_components, EEG.nbchan - 1);
        [~, EEG]= evalc("pop_runica(EEG, 'icatype', 'runica', 'pca', actual_rank, 'extended', 1)");

       [~, EEG]= evalc("pop_iclabel(EEG, 'default')");
        
        % Trích xuất xác suất từ ICLabel
        classes = EEG.etc.ic_classification.ICLabel.classes; % {'Brain','Muscle','Eye','Heart',...}
        probs = EEG.etc.ic_classification.ICLabel.classifications;
        
        eye_idx = find(strcmp(classes, 'Eye'));
        heart_idx = find(strcmp(classes, 'Heart'));
        muscle_idx = find(strcmp(classes, 'Muscle'));
        
        bad_ics = [];
        for c = 1:size(probs, 1)
            % Điều kiện loại bỏ giống hệt script Python của bạn
            if probs(c, eye_idx) > 0.90 || probs(c, heart_idx) > 0.90 || probs(c, muscle_idx) > 0.90
                bad_ics = [bad_ics, c];
            end
        end
        
        if ~isempty(bad_ics)
            fprintf('  -> Loại bỏ %d ICs (Mắt/Tim/Cơ).\n', length(bad_ics));
            [~, EEG] =evalc(" pop_subcomp(EEG, bad_ics, 0)");
        end

        data = EEG.data;
        
        mu = mean(data, 2);        
        sigma = std(data, 0, 2);   
        data_zscored = (data - mu) ./ sigma;
        if size(data_zscored, 2) < 1000
            fprintf('!!! BỎ QUA: Dữ liệu quá ngắn.\n');
            clean_data = []; return;
        end
        if any(std(data_zscored, 0, 2) < 1e-10)
            fprintf('!!! BỎ QUA: Có tín hiệu chết.\n');
            clean_data = []; return;
        end
        if any(isnan(data_zscored(:))) || any(isinf(data_zscored(:)))
            fprintf('!!! BỎ QUA: Chứa NaN hoặc Inf.\n');
            clean_data = []; return;
        end

        % Nếu mọi thứ ổn, gán kết quả đầu ra
        clean_data = data_zscored;
        fprintf('  -> Hoàn tất.\n');

    catch ME
        fprintf('!!! LỖI TẠI FILE %s: %s\n', file_name, ME.message);
        clean_data = [];
    end
end