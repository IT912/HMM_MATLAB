import os
import glob
import shutil
import numpy as np
from scipy.io import savemat
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Model, Config
import scipy.io as sio

def prepare_data_for_hmm(npy_files_list, memmap_dir, n_embed, n_pca):
    """Hàm chuẩn bị dữ liệu: áp dụng Time-Delay Embedding (TDE) và PCA"""
    if os.path.exists(memmap_dir): 
        shutil.rmtree(memmap_dir)
    os.makedirs(memmap_dir)
    
    print(f"[Data] Đang nạp {len(npy_files_list)} file và chuẩn bị TDE({n_embed}) + PCA({n_pca})...")
    data = Data(npy_files_list, store_dir=memmap_dir)
    data.prepare(methods={
        "tde_pca": {"n_embeddings": n_embed, "n_pca_components": n_pca},
        "standardize": {}
    })
    return data

def train_hmm_model(npy_files_list, memmap_dir, n_states, seq_len, batch_size, lr, epochs, n_embed, n_pca, model_save_dir):
    """Huấn luyện mô hình HMM và lưu lại mô hình tốt nhất"""
    # Chuẩn bị dữ liệu
    training_data = prepare_data_for_hmm(npy_files_list, memmap_dir, n_embed, n_pca)
    
    # Cấu hình mạng HMM
    config = Config(
        n_states=n_states,
        n_channels=training_data.n_channels,
        sequence_length=seq_len,
        learn_means=False, # Thường đặt False nếu đã dùng zero-mean/standardize
        learn_covariances=True, 
        batch_size=batch_size,
        learning_rate=lr,
        n_epochs=epochs 
    )

    n_runs = 1 
    best_fe = float('inf')
    best_model = None

    for run_i in range(n_runs):
        print(f"\n--- RUN {run_i + 1}/{n_runs} ---")
        model = Model(config)
        # Khởi tạo chuỗi trạng thái ngẫu nhiên ban đầu
        model.random_state_time_course_initialization(training_data, n_epochs=1, n_init=3)
        # Huấn luyện
        model.fit(training_data)
        
        # Đánh giá dựa trên Free Energy (càng thấp càng tốt)
        fe = model.free_energy(training_data)
        if fe < best_fe:
            best_fe = fe
            best_model = model

    print(f"\n[Save] Đang lưu mô hình tốt nhất vào: {model_save_dir}")
    if os.path.exists(model_save_dir): 
        shutil.rmtree(model_save_dir)
    best_model.save(model_save_dir)
    return best_model, training_data

def export_for_matlab(model, training_data, output_mat_path):
    print(f"\n[Export] Đang trích xuất dữ liệu để đưa về MATLAB...")
    alpha = model.get_alpha(training_data)
    viterbi_paths = [np.argmax(a, axis=1) for a in alpha]
    trans_prob = model.get_trans_prob()
    means, covs = model.get_means_covariances()
    matlab_data = {
        'transition_matrix': trans_prob,
        'state_means': means,
        'state_covariances': covs,
        'viterbi_paths': np.array(viterbi_paths, dtype=object), # Lưu dưới dạng cell array
        'state_probabilities': np.array(alpha, dtype=object)
    }
    
    savemat(output_mat_path, matlab_data)
    print(f"[Done] Đã lưu file kết quả MATLAB tại: {output_mat_path}")

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    BASE_DIR = "/home/dung/CODE_MATLAB"
    TEMP_NPY_DIR = os.path.join(BASE_DIR, "temp_npy")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "model_save")
    MEMMAP_DIR = os.path.join(BASE_DIR, "memmap")
    
    # SỬA LỖI 3: Trỏ đường dẫn vào thẳng một file .mat bên trong thư mục results
    MATLAB_EXPORT_FILE = os.path.join(BASE_DIR, "results", "hmm_results.mat")

    # Lấy danh sách tất cả các file .mat
    mat_files = sorted(
    glob.glob(os.path.join(TEMP_NPY_DIR, "*.mat")), 
    key=lambda x: int(os.path.basename(x).replace('sub_', '').replace('.mat', ''))
)
    if not mat_files:
        raise ValueError(f"Không tìm thấy file .mat nào trong {TEMP_NPY_DIR}. Vui lòng kiểm tra lại!")
    VARIABLE_NAME = 'data' 
    numpy_data_list = []
    print(f"[Load] Đang trích xuất và xoay chiều dữ liệu từ {len(mat_files)} file .mat...")
    
    for file in mat_files:
        mat_contents = sio.loadmat(file)
        
        # Bắt lỗi nếu gõ sai tên biến
        if VARIABLE_NAME not in mat_contents:
            available_vars = [k for k in mat_contents.keys() if not k.startswith('__')]
            raise KeyError(f"Không tìm thấy biến '{VARIABLE_NAME}' trong {file}. Các biến hiện có: {available_vars}")
            
        eeg_matrix = mat_contents[VARIABLE_NAME]
        
        # Xoay ma trận nếu MATLAB đang lưu theo chuẩn [Channels x Time]
        if eeg_matrix.shape[0] < eeg_matrix.shape[1]:
            eeg_matrix = eeg_matrix.T 
            
        numpy_data_list.append(eeg_matrix)
    
    N_STATES = 8             
    N_EMBED = 15             
    N_PCA = 40               
    SEQ_LEN = 200            
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 20             

    # 1. Tiến hành huấn luyện (Truyền numpy_data_list thay vì mat_files)
    best_model, processed_data = train_hmm_model(
        npy_files_list=numpy_data_list,
        memmap_dir=MEMMAP_DIR,
        n_states=N_STATES,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        n_embed=N_EMBED,
        n_pca=N_PCA,
        model_save_dir=MODEL_SAVE_DIR
    )

    # 2. Xuất dữ liệu sang định dạng MATLAB
    export_for_matlab(best_model, processed_data, MATLAB_EXPORT_FILE)
    
    if os.path.exists(MEMMAP_DIR):
        shutil.rmtree(MEMMAP_DIR)