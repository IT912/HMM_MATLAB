% ==========================================
% CÔNG TẮC ĐIỀU KHIỂN QUY TRÌNH
% ==========================================
RUN_PREPROCESSING = false ;

fprintf('=== KHỞI ĐỘNG HỆ THỐNG PHÂN TÍCH EEG (MATLAB) ===\n');
%eeglab nogui;

% 1. Nạp cấu hình từ file config_eeg.m
cfg = config_eeg();

% Khởi tạo cell array chứa dữ liệu đưa vào HMM
eeg_dataset = {}; 

if RUN_PREPROCESSING
    fprintf('\n--- BƯỚC 1 & 2: Quét thư mục và Tiền xử lý dữ liệu ---\n');
    
    if ~exist(cfg.npy_dir, 'dir')
        mkdir(cfg.npy_dir); 
    end
    
    group_folders = {'intact_cognition', 'decreased_cognition'};
    valid_file_count = 1;
    
    for g = 1:length(group_folders)
        current_folder = fullfile(cfg.base_dir, group_folders{g});
        files = dir(fullfile(current_folder, '*.set')); 
        
        for i = 1:length(files)
            fpath = fullfile(current_folder, files(i).name);
            
            clean_data = preprocess_eeg_data(fpath, cfg);
            
            if ~isempty(clean_data)
                % --- ĐÃ SỬA LỖI 1 ---
                data = clean_data; % Đổi tên biến thành 'data' để khớp với Python
                save_name = sprintf('sub_%d.mat', valid_file_count);
                save_path = fullfile(cfg.npy_dir, save_name);
                
                % Chỉ lưu biến 'data' vào đúng đường dẫn save_path
                save(save_path, 'data'); 
                
                eeg_dataset{valid_file_count} = data;
                valid_file_count = valid_file_count + 1;
            end
        end
    end
    fprintf('Đã tiền xử lý và lưu %d file hợp lệ vào temp_npy.\n', length(eeg_dataset));
    
else
    fprintf('\n--- BỎ QUA BƯỚC 1 & 2: Nạp dữ liệu sẵn có ---\n');
    n_files_to_load = 60; 
    eeg_dataset = cell(1, n_files_to_load);
    
    for i = 1:n_files_to_load
        fpath = fullfile(cfg.npy_dir, sprintf('sub_%d.mat', i));
        
        if exist(fpath, 'file')
            temp = load(fpath); 
            eeg_dataset{i} = temp.data; 
        else
            error('!!! LỖI: Không tìm thấy file %s. Dữ liệu bị thiếu!', fpath);
        end
    end
    fprintf('Đã nạp %d file dữ liệu thành công.\n', length(eeg_dataset));
end

% ==========================================
% --- ĐÃ SỬA LỖI 2: KẾT NỐI VỚI KẾT QUẢ TỪ PYTHON ---
% ==========================================
fprintf('\n--- BƯỚC 3: Nạp kết quả HMM từ Python ---\n');

% Trỏ tới file kết quả mà Python xuất ra (kiểm tra lại thư mục results có tồn tại không)
python_results_path = fullfile('D:\MATLAB\', 'results', 'hmm_results.mat');

if ~exist(python_results_path, 'file')
    fprintf('!!! CHÚ Ý: Chưa có file hmm_results.mat từ Python.\n');
    fprintf('=> Hãy tạm dừng MATLAB tại đây. Mở Terminal/VSCode và chạy file train_model.py.\n');
    fprintf('=> Sau khi Python chạy xong, hãy chạy tiếp phần code bên dưới.\n');
    return; % Tạm dừng chương trình tại đây chờ bạn chạy Python
else
    fprintf('Đã tìm thấy file kết quả từ Python! Đang nạp...\n');
    hmm_results = load(python_results_path);
    
    % Gán kết quả vào các biến tương đương để đưa vào hàm generate_report
    % (Tuỳ thuộc vào hàm generate_report của bạn đang cần biến gì)
    best_hmm = hmm_results; 
    best_Gamma = hmm_results.state_probabilities;
end

fprintf('\n--- BƯỚC 4: Trực quan hóa & Xuất báo cáo Tổng thể ---\n');
try
    generate_report(best_hmm, best_Gamma, eeg_dataset, cfg);
    fprintf('\n=== HOÀN TẤT TOÀN BỘ QUY TRÌNH ===\n');
catch ME
    fprintf('!!! LỖI TẠI BƯỚC VẼ BIỂU ĐỒ: %s\n', ME.message);
end