config = {
    "preprocessing":
        {
            "remove_columns_ending_with":
                (
                    "code",  # is a coded score and not of interest for the case
                    "procedure",  # is the same for the hip or knee set
                    "revision_flag",  # revisions are out of scope, filtered away, so same for all rows after that
                    "assisted_by",  # is the same for all records
                    "profile",  # is a coded score and not of interest for the case
                    "predicted",  # are predictions of other models that are not supposed to be used
                ),
            "remove_low_info_categories":
                {
                    "t0_assisted": "no",
                    "t0_previous_surgery": "no",
                    "t0_disability": "no",
                }
        },
    "outputs":
        {
            "hip": {
                "t1_ohs_score": {
                    "bins":[0, 19, 29, 39, 48],
                    "labels":["severe", "moderate", "mild-moderate", "satisfactory"],
                },
                "t1_eq5d_mobility":{
                    "labels":["no problems", "some problems", "severe problems"],
                },
                # "t1_eq5d_self_care":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
                # "t1_eq5d_activity":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
                # "t1_eq5d_discomfort":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
                # "t1_eq5d_anxiety":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
            },
            "knee": {
                "t1_oks_score": {
                    "bins":[0, 19, 29, 39, 48],
                    "labels":["severe", "moderate", "mild-moderate", "satisfactory"],
                },
                "t1_eq5d_mobility":{
                    "labels":["no problems", "some problems", "severe problems"],
                },
                # "t1_eq5d_self_care":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
                # "t1_eq5d_activity":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
                # "t1_eq5d_discomfort":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
                # "t1_eq5d_anxiety":{
                #     "labels":["no problems", "some problems", "severe problems"],
                # },
            },
        },
    "score":"roc_auc_ovo_weighted",
    "models":{
        "path":"models",
    }
}
