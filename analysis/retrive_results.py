from src.load_results import model_selection_parallel as model_selection


FairSCL_df = model_selection(
    results_dir= r"/data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification",
    project_dir= r"FairSCL/Bios_gender",
    checkpoint_dir= "models",
    checkpoint_name= "checkpoint_epoch",
    model_id= ("FSCL"),
    GAP_metric_name = "TPR_GAP",
    Performance_metric_name = "accuracy",
    selection_criterion = "DTO",
    n_jobs=20,
    index_column_names = ["fcl_lambda_y"],
    save_path = "results/Bios_gender_FairSCL_df.pkl"
    )

GDEO_df = model_selection(
    results_dir= r"/data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification",
    project_dir= r"GroupDifference/Bios_gender",
    checkpoint_dir= "models",
    checkpoint_name= "checkpoint_epoch",
    model_id= ("GDEO"),
    GAP_metric_name = "TPR_GAP",
    Performance_metric_name = "accuracy",
    selection_criterion = "DTO",
    n_jobs=20,
    index_column_names = ["DyBTalpha"],
    save_path = "results/Bios_gender_GDEO_df.pkl"
    )

GDMean_df = model_selection(
    results_dir= r"/data/cephfs/punim0478/xudongh1/experimental_results/Fair_NLP_Classification",
    project_dir= r"GroupDifference/Bios_gender",
    checkpoint_dir= "models",
    checkpoint_name= "checkpoint_epoch",
    model_id= ("GDMean"),
    GAP_metric_name = "TPR_GAP",
    Performance_metric_name = "accuracy",
    selection_criterion = "DTO",
    n_jobs=20,
    index_column_names = ["DyBTalpha"],
    save_path = "results/Bios_gender_GDMean_df.pkl"
    )