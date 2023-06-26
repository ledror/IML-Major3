def prepare_data(training_data, new_data):
    import pandas as pd

    training_data = training_data.copy()
    out_data = new_data.copy()

    # make blood_type into blood_type_A, blood_type_AB, blood_type_O
    prepare_blood_type(training_data)
    prepare_blood_type(out_data)

    # split symptoms into each symptom
    prepare_symptoms(training_data)
    prepare_symptoms(out_data)

    # make sex -> male
    prepare_male(training_data)
    prepare_male(out_data)

    # current_location -> x, y
    prepare_location(training_data)
    prepare_location(out_data)

    # pcr_date -> days since 1/1/2020
    prepare_pcr_date(training_data)
    prepare_pcr_date(out_data)

    to_min_max_scale = ["patient_id",
                        "age",
                        "sport_activity",
                        "pcr_date",
                        "PCR_01",
                        "PCR_02",
                        "PCR_03",
                        "PCR_04",
                        "PCR_05",
                        "PCR_07",
                        "PCR_09",
                        "PCR_10",
                        "x_location",
                        "y_location"]

    to_standard_scale = ["weight",
                        "num_of_siblings",
                        "happiness_score",
                        "household_income",
                        "conversations_per_day",
                        "sugar_levels",
                        "PCR_06",
                        "PCR_08"]
    
    # fit to training_data, transform out_data
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    min_max_scaler = MinMaxScaler((-1, 1))
    min_max_scaler.fit(training_data[to_min_max_scale])
    out_data[to_min_max_scale] = min_max_scaler.transform(out_data[to_min_max_scale])

    standard_scaler = StandardScaler()
    standard_scaler.fit(training_data[to_standard_scale])
    out_data[to_standard_scale] = standard_scaler.transform(out_data[to_standard_scale])

    return out_data

def prepare_blood_type(data):
    data["blood_type_A"] = data["blood_type"].isin(["A+", "A-"]).astype(int)
    data["blood_type_AB"] = data["blood_type"].isin(["AB+", "AB-", "B+", "B-"]).astype(int)
    data["blood_type_O"] = data["blood_type"].isin(["O+", "O-"]).astype(int)
    data.drop("blood_type", axis=1, inplace=True)
    return data

def prepare_symptoms(data):
    data["symptoms"].fillna("", inplace=True)
    allSymptoms = set()
    for symptoms in data["symptoms"].unique():
        allSymptoms.update(symptoms.split(";"))
    allSymptoms.remove("")
    for symptom in allSymptoms:
        data[symptom] = data["symptoms"].str.contains(symptom).map({True: 1, False: 0})
    data.drop("symptoms", axis=1, inplace=True)
    return data

def prepare_male(data):
    data.rename({"sex" : "male"}, axis=1, inplace=True)
    data["male"].replace({"M" : 1, "F" : 0}, inplace=True)
    return data

def prepare_location(data):
    data["x_location"] = data["current_location"].str.split(",").str[0]
    data["y_location"] = data["current_location"].str.split(",").str[1]
    data["x_location"] = data["x_location"].str.replace("(", "").str.replace("'", "").astype(float)
    data["y_location"] = data["y_location"].str.replace(")", "").str.replace("'", "").astype(float)
    data.drop("current_location", axis=1, inplace=True)
    return data

def prepare_pcr_date(data):
    import pandas as pd
    data["pcr_date"] = pd.to_datetime(data["pcr_date"])
    data["pcr_date"] = (data["pcr_date"] - pd.to_datetime("2020-01-01")).dt.days
    return data