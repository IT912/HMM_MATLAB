function cfg = config_eeg()
    
    cfg = struct();
    cfg.base_dir = 'D:\MATLAB\MS_EEG';
    cfg.npy_dir = 'D:\MATLAB\temp_npy';
    cfg.memmap_dir = 'D:\MATLAB\memmap';
    cfg.model_save = 'D:\MATLAB\model_npy';

    
    cfg.target_sfreq = 250;           
    cfg.ica_components = 19;          
    cfg.bandpass = [1, 45];          
    cfg.notch_freqs = [50, 100];     
    cfg.random_state = 42;           

    cfg.n_embeddings = 15;
    cfg.n_pca_components = 0;

    cfg.n_states = 8;
    cfg.seq_length = 800;            
    cfg.batch_size = 64;
    cfg.learning_rate = 0.01;
    cfg.n_epochs = 1;

    cfg.freq_bands.Delta = [1, 4];
    cfg.freq_bands.Theta = [4, 8];
    cfg.freq_bands.Alpha = [8, 13];
    cfg.freq_bands.Beta  = [13, 30];
end