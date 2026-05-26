function generate_report(best_hmm, best_Gamma, eeg_dataset, cfg)
    % Hàm trực quan hóa tương thích hoàn toàn với main_pipeline

    output_dir = fullfile(cfg.base_dir, 'EEG_Reports');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % ==============================================================
    % BƯỚC 0: TỰ ĐỘNG XÁC ĐỊNH SỐ LƯỢNG NHÓM (ĐÃ SỬA LỖI CẢNH BÁO)
    % ==============================================================
    fprintf('\n[BÁO CÁO] 0. Đang phân bổ dữ liệu vào các nhóm...\n');
    
    % Chia đôi danh sách file đã nạp thay vì quét lại thư mục
    half_size = floor(length(eeg_dataset) / 2);
    group_count = [half_size, length(eeg_dataset) - half_size];
    group_names = {'Initial Cog', 'Decreased Cog'};

    n_states = cfg.n_states;
    fs = cfg.target_sfreq;
    n_subs = length(eeg_dataset);
    
    % ==============================================================
    % BƯỚC 1: TRÍCH XUẤT VÀ ĐỒNG BỘ CHIỀU DÀI DỮ LIỆU (FIX LỖI INDEX)
    % ==============================================================
    fprintf('[BÁO CÁO] 1. Đang đồng bộ chiều dài dữ liệu (Bù trừ TDE)...\n');
    
    vpath_all = cell(n_subs, 1);
    data_aligned = cell(n_subs, 1);
    
    for s = 1:n_subs
        % 1. Lấy xác suất trạng thái của từng người
        gamma_s = best_Gamma{s};
        
        % Đảm bảo ma trận dọc [Time x States]
        if size(gamma_s, 2) > size(gamma_s, 1)
            gamma_s = gamma_s';
        end
        
        % 2. Tìm Viterbi path (Trạng thái trội nhất)
        [~, vp] = max(gamma_s, [], 2);
        vpath_all{s} = vp'; % [1 x Time]
        
        % 3. Cắt tín hiệu gốc cho khớp với chiều dài của Viterbi path
        d_s = eeg_dataset{s};
        diff_len = size(d_s, 2) - length(vp);
        
        if diff_len > 0
            % TDE cắt ở 2 đầu, ta cũng cắt đều 2 đầu của data_s
            trim_start = floor(diff_len / 2) + 1;
            trim_end = trim_start + length(vp) - 1;
            data_aligned{s} = d_s(:, trim_start:trim_end);
        else
            data_aligned{s} = d_s;
        end
    end

    % ==============================================================
    % BƯỚC 2: TÍNH TOÁN RELATIVE PSD BẰNG FFT (KHÔNG CẦN TOOLBOX)
    % ==============================================================
    fprintf('[BÁO CÁO] 2. Đang tính toán Multitaper Spectra (FFT)...\n');
    
    nfft = 500; % Window length
    f = fs * (0:(nfft/2)) / nfft;
    valid_f_idx = find(f >= cfg.bandpass(1) & f <= cfg.bandpass(2));
    f_plot = f(valid_f_idx);
    
    relative_psd = zeros(n_subs, n_states, length(f_plot));

    for s = 1:n_subs
        data_s = data_aligned{s}; % Đã dùng dữ liệu đã đồng bộ
        vpath_s = vpath_all{s};
        
        % PSD toàn bộ tín hiệu
        L_all = size(data_s', 1);
        Y_all = fft(data_s', nfft, 1);
        P2_all = abs(Y_all/L_all);
        P1_all = P2_all(1:floor(nfft/2)+1, :);
        P1_all(2:end-1, :) = 2 * P1_all(2:end-1, :);
        pxx_all = P1_all.^2; 
        
        mean_psd_all = mean(pxx_all(valid_f_idx, :), 2)'; 
        
        % PSD cho từng trạng thái
        for k = 1:n_states
            state_time_points = find(vpath_s == k);
            if length(state_time_points) > nfft
                data_state_k = data_s(:, state_time_points);
                
                L_k = size(data_state_k', 1);
                Y_k = fft(data_state_k', nfft, 1);
                P2_k = abs(Y_k/L_k);
                P1_k = P2_k(1:floor(nfft/2)+1, :);
                P1_k(2:end-1, :) = 2 * P1_k(2:end-1, :);
                pxx_k = P1_k.^2;
                
                mean_psd_k = mean(pxx_k(valid_f_idx, :), 2)';
                relative_psd(s, k, :) = mean_psd_k - mean_psd_all;
            else
                relative_psd(s, k, :) = 0; 
            end
        end
    end

    % ==============================================================
    % BƯỚC 3: VẼ BIỂU ĐỒ RELATIVE PSD CHO TỪNG NHÓM
    % ==============================================================
    fprintf('[BÁO CÁO] 3. Đang vẽ biểu đồ Relative PSD...\n');
    
    start_idx = 1;
    for g = 1:length(group_count)
        if group_count(g) == 0, continue; end
        end_idx = min(start_idx + group_count(g) - 1, n_subs);
        
        group_rpsd = relative_psd(start_idx:end_idx, :, :); 
        group_rpsd_mean = squeeze(mean(group_rpsd, 1)); 
        
        fig = figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
        hold on;
        colors = lines(n_states); 
        
        for k = 1:n_states
            plot(f_plot, group_rpsd_mean(k, :), 'LineWidth', 2, 'Color', colors(k,:), 'DisplayName', sprintf('State %d', k));
        end
        
        yline(0, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off'); 
        title(sprintf('Relative PSD - %s', group_names{g}), 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Frequency (Hz)', 'FontSize', 12);
        ylabel('Power relative to global mean', 'FontSize', 12);
        xlim([cfg.bandpass(1), cfg.bandpass(2)]);
        legend('Location', 'bestoutside');
        grid on; hold off;
        
        file_name = sprintf('hmm_states_relative_psd_%s.png', strrep(group_names{g}, ' ', '_'));
        exportgraphics(fig, fullfile(output_dir, file_name), 'Resolution', 300);
        close(fig);
        
        start_idx = end_idx + 1;
    end

    % ==============================================================
    % BƯỚC 4: TÍNH TOÁN ĐỘNG LỰC HỌC (FRACTIONAL OCCUPANCY & MEAN LIFE)
    % ==============================================================
    fprintf('\n[ĐỘNG LỰC HỌC] 1. Đang tính toán FO và Mean Life...\n');
    
    FO = zeros(n_subs, n_states);
    MeanLife = zeros(n_subs, n_states);
    
    for s = 1:n_subs
        vpath_s = vpath_all{s};
        T_s = length(vpath_s); % Dùng chiều dài đã chuẩn hóa
        
        for k = 1:n_states
            state_mask = (vpath_s == k);
            FO(s, k) = sum(state_mask) / T_s * 100;
            
            transitions = diff([0, state_mask, 0]); 
            starts = find(transitions == 1);
            ends = find(transitions == -1);
            
            durations_samples = ends - starts; 
            
            if isempty(durations_samples)
                MeanLife(s, k) = 0;
            else
                MeanLife(s, k) = mean(durations_samples) * (1000 / fs);
            end
        end
    end

    fprintf('[ĐỘNG LỰC HỌC] 2. Đang xuất biểu đồ...\n');
    
    FO_means = zeros(n_states, 2); FO_sems  = zeros(n_states, 2);
    ML_means = zeros(n_states, 2); ML_sems  = zeros(n_states, 2);
    
    start_idx = 1;
    for g = 1:2
        end_idx = min(start_idx + group_count(g) - 1, n_subs);
        
        group_FO = FO(start_idx:end_idx, :);
        group_ML = MeanLife(start_idx:end_idx, :);
        
        FO_means(:, g) = mean(group_FO, 1)';
        ML_means(:, g) = mean(group_ML, 1)';
        
        N = size(group_FO, 1);
        FO_sems(:, g) = (std(group_FO, 0, 1) / sqrt(N))';
        ML_sems(:, g) = (std(group_ML, 0, 1) / sqrt(N))';
        
        start_idx = end_idx + 1;
    end

    % BIỂU ĐỒ FO
    fig1 = figure('Visible', 'off', 'Position', [100, 100, 900, 500]);
    b1 = bar(FO_means, 'grouped');
    b1(1).FaceColor = [0.2 0.6 0.8]; b1(2).FaceColor = [0.8 0.3 0.3]; 
    hold on;
    for i = 1:2
        errorbar(b1(i).XEndPoints, FO_means(:, i), FO_sems(:, i), 'k', 'linestyle', 'none', 'lineWidth', 1.2);
    end
    hold off;
    title('Fractional Occupancy', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('HMM States', 'FontSize', 12); ylabel('Time Spent (%)', 'FontSize', 12);
    xticks(1:n_states); xticklabels(arrayfun(@(x) sprintf('State %d', x), 1:n_states, 'UniformOutput', false));
    legend(group_names, 'Location', 'northwest'); grid on;
    exportgraphics(fig1, fullfile(output_dir, 'temporal_Fractional_Occupancy.png'), 'Resolution', 300);
    close(fig1);

    % BIỂU ĐỒ MEAN LIFE
    fig2 = figure('Visible', 'off', 'Position', [150, 150, 900, 500]);
    b2 = bar(ML_means, 'grouped');
    b2(1).FaceColor = [0.2 0.6 0.8]; b2(2).FaceColor = [0.8 0.3 0.3];
    hold on;
    for i = 1:2
        errorbar(b2(i).XEndPoints, ML_means(:, i), ML_sems(:, i), 'k', 'linestyle', 'none', 'lineWidth', 1.2);
    end
    hold off;
    title('Mean Life', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('HMM States', 'FontSize', 12); ylabel('Duration (ms)', 'FontSize', 12);
    xticks(1:n_states); xticklabels(arrayfun(@(x) sprintf('State %d', x), 1:n_states, 'UniformOutput', false));
    legend(group_names, 'Location', 'northwest'); grid on;
    exportgraphics(fig2, fullfile(output_dir, 'temporal_Mean_Life.png'), 'Resolution', 300);
    close(fig2);
    
    fprintf('=== ĐÃ LƯU BÁO CÁO TẠI: %s ===\n', output_dir);
end