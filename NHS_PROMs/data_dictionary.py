import itertools

# dict with method info
methods = dict(
    eq5d=dict(
        dims=dict(
            names=("mobility", "self_care", "activity", "discomfort", "anxiety"),
            labels={
                1: "no problems",
                2: "some problems",
                3: "severe problems",
                9: "missing",
            },
            kind="ordinal",
        ),
        profile={
            "labels": {
                int(c): f"profile {c}"
                for c in ["".join(c) for c in itertools.product("123", repeat=5)]
            },
            "kind": "ordinal",
        },
        score={"range": (-0.594, 1), "kind": "numerical"},
        predicted={"kind":"numerical"}
    ),
    ohs=dict(
        dims=dict(
            names=(
                "pain",
                "sudden_pain",
                "night_pain",
                "washing",
                "transport",
                "dressing",
                "shopping",
                "walking",
                "limping",
                "stairs",
                "standing",
                "work",
            ),
            labels={
                0: "all of the time",
                1: "most of the time",
                2: "often, not just at first",
                3: "sometimes or just at first",
                4: "rarely/never",
                9: "missing",
            },
            kind="ordinal",
        ),
        score={"range": (0, 48), "kind": "numerical"},
        predicted={"kind":"numerical"},
    ),
    oks=dict(
        dims=dict(
            names=(
                "pain",
                "night_pain",
                "washing",
                "transport",
                "walking",
                "standing",
                "limping",
                "kneeling",
                "work",
                "confidence",
                "shopping",
                "stairs",
            ),
            labels={
                0: "all of the time",
                1: "most of the time",
                2: "often, not just at first",
                3: "sometimes or just at first",
                4: "rarely/never",
                9: "missing",
            },
            kind="ordinal",
        ),
        score={"range": (0, 48), "kind": "numerical"},
        predicted={"kind":"numerical"},
    ),
    eqvas={"score": {"range": (0, 100), "kind": "numerical"},
           "predicted": {"kind": "numerical"}},
)

# dicts for other columns
demographics = dict(
    age_band=dict(
        labels={
            "20 to 29": "20 to 29",
            "30 to 39": "30 to 39",
            "40 to 49": "40 to 49",
            "50 to 59": "50 to 59",
            "60 to 69": "60 to 69",
            "70 to 79": "70 to 79",
            "80 to 89": "80 to 89",
            "90 to 120": "90 to 120",
        },
        kind="ordinal",
    ),
    gender=dict(
        labels={0: "not known", 1: "male", 2: "female", 9: "missing"},
        kind="categorical",
    ),
    living_arrangements=dict(
        labels={
            1: "with partner / spouse / family / friends",
            2: "alone",
            3: "in a nursing home, hospital or other long-term care home",
            4: "other",
            9: "missing",
        },
        kind="categorical",
    ),
)

comorbidities = {
    c: dict(labels={1: "yes", 9: "missing"}, kind="categorical")
    for c in [
        "heart_disease",
        "high_bp",
        "stroke",
        "circulation",
        "lung_disease",
        "diabetes",
        "kidney_disease",
        "nervous_system",
        "liver_disease",
        "cancer",
        "depression",
        "arthritis",
    ]
}
disability = {"disability": dict(labels={1: "yes", 2: "no", 9: "missing"}, kind="categorical")}

procedure = dict(
    procedure=dict(
        labels={
            "Hip Replacement": "hip Replacement",
            "Knee Replacement": "knee Replacement",
        },
        kind="categorical",
    ),
    revision_flag=dict(
        labels={0: "no revision", 1: "revision procedure"}, kind="categorical"
    ),
    previous_surgery=dict(labels={1: "yes", 2: "no", 9: "missing"}, kind="categorical"),
    symptom_period=dict(
        labels={
            1: "less than 1 year",
            2: "1 to 5 years",
            3: "6 to 10 years",
            4: "more than 10 years",
            9: "missing",
        },
        kind="ordinal",
    ),
)

complications = {
    i: dict(labels={1: "yes", 2: "no", 9: "missing"}, kind="categorical")
    for i in ["allergy", "urine", "bleeding", "wound", "readmitted", "further_surgery"]
}

result = dict(
    satisfaction=dict(
        labels={
            1: "excellent",
            2: "very good",
            3: "good",
            4: "fair",
            5: "poor",
            9: "missing",
        },
        kind="ordinal",
    ),
    success=dict(
        labels={
            1: "much better",
            2: "a little better",
            3: "about the same",
            4: "a little worse",
            5: "much worse",
            9: "missing",
        },
        kind="ordinal",
    ),
)

registration = dict(
    assisted=dict(labels={1: "yes", 2: "no", 9: "missing",}, kind="categorical"),
    assisted_by=dict(
        labels={
            1: "family member (e.g. spouse, child, parent)",
            2: "other relative",
            3: "carer",
            4: "friend/neighbour",
            5: "healthcare professional (e.g. nurse/doctor)",
            6: "other",
            9: "missing",
        },
        kind="categorical",
    ),
)

other = dict(
    provider_code={"kind":"categorical"},
    year={"kind":"ordinal", "labels":{"2016/17": "April 2016 - April 2017", "2017/18": "April 2017 - April 2018", "2018/19": "April 2018 - April 2019", "2019/20": "April 2019 - April 2020"}},
)

# unpack methods meta
method_meta = dict()
for method, meta in methods.items():
    if meta.get("dims"):
        for dim in meta["dims"].get("names"):
            temp = dict()
            for key in ["labels", "range", "kind"]:
                if meta["dims"].get(key):
                    temp.update({key: meta["dims"][key]})
            method_meta.update({f"{method}_{dim}":temp})
    for agg in ["score", "profile", "predicted"]:
        if meta.get(agg):
            temp = dict()
            for key in ["labels", "range", "kind"]:
                    if meta[agg].get(key):
                        temp.update({key: meta[agg][key]})
            method_meta.update({f"{method}_{agg}":temp})

meta_dict = {
    **method_meta,
    **demographics,
    **comorbidities,
    **disability,
    **procedure,
    **complications,
    **result,
    **registration,
    **other,
}