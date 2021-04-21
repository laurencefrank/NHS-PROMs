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
                )
        },
    "outputs":
        {
            "hip": {
                "t1_ohs_score": {
                    "bins":[0, 19, 29, 39, 48],
                    "labels":["severe", "moderate", "mild-moderate", "satisfactory"],
                },
                "t1_eq5d_mobility":None,
                "t1_eq5d_self_care":None,
                "t1_eq5d_activity":None,
                "t1_eq5d_discomfort":None,
                "t1_eq5d_anxiety":None,
            },
            "knee": {
                "t1_oks_score": {
                    "bins":[0, 19, 29, 39, 48],
                    "labels":["severe", "moderate", "mild-moderate", "satisfactory"],
                },
                "t1_eq5d_mobility":None,
                "t1_eq5d_self_care":None,
                "t1_eq5d_activity":None,
                "t1_eq5d_discomfort":None,
                "t1_eq5d_anxiety":None,
            },
        },
    "score":"roc_auc_ovo_weighted",
    "models":{
        "path":"models",
    }
}
