from src.load_results import model_selection_parallel as model_selection


FairSCL_df = model_selection(
    results_dir= r"/data/cephfs/punim1421/Fair_NLP_Classification/results",
    project_dir= r"hypertune/Moji",
    checkpoint_dir= "models",
    checkpoint_name= "checkpoint_epoch",
    model_id= ("DelayedCLS_Adv"),
    GAP_metric_name = "TPR_GAP",
    Performance_metric_name = "accuracy",
    selection_criterion = "DTO",
    n_jobs=20,
    index_column_names = ["classification_head_update_frequency", "adv_lambda"],
    save_path = "results/Moji_DelayedCLS_Adv_df.pkl"
    )