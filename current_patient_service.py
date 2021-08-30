import custom_shap as cshap
import pandas as pd
from flask import jsonify
import pickle
import shap
from custom_shap import summary_with_highlight 
import uuid
import base64
import os
from PIL import Image as img
from matplotlib import pyplot as plt
import matplotlib.pyplot as pl; 


class PatientFullPrediction():
    def __init__(self, session_id, patient_id, cardiac_proba, cardiac_lim, pulmonary_proba,
                 pulmonary_lim, other_proba, other_lim) -> None:
        self.session_id = int(session_id)
        self.patient_id = float(patient_id)
        self.cardiac_proba = int(cardiac_proba*100)
        self.cardiac_lim = int(cardiac_lim*100)
        self.pulmonary_proba = int(pulmonary_proba*100)
        self.pulmonary_lim = int(pulmonary_lim*100)
        self.other_proba = int(other_proba*100)
        self.other_lim = int(other_lim*100)
        pass
    pass

class PatientDynamicFullPrediction():
    def __init__(self, session_id, patient_id, cardiac_proba_array, cardiac_lim, pulmonary_proba_array,
                 pulmonary_lim, other_proba_array, other_lim) -> None:
        self.session_id = int(session_id)
        self.patient_id = float(patient_id)
        self.cardiac_lim = int(cardiac_lim*100)
        self.pulmonary_lim = int(pulmonary_lim*100)
        self.other_lim = int(other_lim*100)
        self.cardiac_proba = cardiac_proba_array
        self.pulmonary_proba = pulmonary_proba_array
        self.other_proba = other_proba_array
        self.time_list = [40, 50, 60, 70, 80, 90, 100]
        pass
    pass


class CPETInterpretationImages():
    def __init__(self, cardiac_summary, cardiac_force, pulmonary_summary, pulmonary_force, 
                other_summary, other_force) -> None:
        self.cardiac_summary = cardiac_summary
        self.cardiac_force = cardiac_force
        self.pulmonary_summary = pulmonary_summary
        self.pulmonary_force = pulmonary_force
        self.other_summary = other_summary
        self.other_force = other_force
        pass


def _generate_array(df, type_lim):
    result = []
    time_list = [40, 50, 60, 70, 80, 90, 100]
    for time in time_list:
        result.append(int(df[type_lim+'LimProba_'+str(time)].values[0]*100))
    return result
    pass


cardiac_data_100 = ['CardiacLim','DiffPercentPeakVO2', 'DiffPeakVO2','75_to_100_VO2Slope','75_to_100_HRSlope','MinO2Pulse',
                      'PeakVE','VO2vsPeakVO2atVT','second_half_RRSlope','second_half_VO2Slope','75_to_100_VCO2Slope','MeanVE',
                      'second_half_VESlope','O2PulseDiff','50_to_75_O2Slope',
                        'O2PulsePercent','75_to_100_RERSlope','PeakRER','50_to_75_VO2Slope','PeakVO2Real']
pulmonary_data_100 = ['PulmonaryLim','O2PulsePercent', 'O2PulseDiff','first_half_VO2Slope','LowestVE/VCO2',
                      'first_half_VCO2Slope', '15_to_85_RRSlope','PeakRR','50_to_75_RRSlope','MeanO2Pulse','VEvsVCO2Slope',
                     '25_to_50_VCO2Slope','StdHeartRate']
other_data_100 = ['MuscleSkeletalLim','PeakRR', 'PeakVE','PeakVCO2','MeanVCO2','PeakVO2','PeakVO2Real',
                  'LowestVE/VCO2','MeanRER','PeakRER','VO2vsPeakVO2atVT','DiffPercentPeakVO2','MeanRR',
                  '75_to_100_VEVCO2Slope','DiffPeakVO2','MeanVE','second_half_VESlope','first_half_VEVCO2Slope',
                  '0_to_25_O2Slope','VO2atVT', 'MeanVO2','second_half_VCO2Slope','DiffPeakHR','MeanVE/VCO2','75_to_100_RRSlope']
cardiac_feature_dict = {
  "DiffPercentPeakVO2": "Actual/Expected Peak VO2",
  "DiffPeakVO2": "Actual - Expected Peak VO2",
  "75_to_100_VO2Slope": "Session's last quarter VO2 slope",
  "75_to_100_HRSlope": "Session's last quarter HR slope",
  "MinO2Pulse": "Minimum O2 pulse",
  "PeakVE": "Max minute ventilation",
  "VO2vsPeakVO2atVT": "VO2 at ventilatory threshold vs expected VO2",
  "second_half_RRSlope": "Second half Respiratory Rate slope",
  "second_half_VO2Slope": "Second half VO2 slope",
  "75_to_100_VCO2Slope": "Last quarter VCO2 slope",
  "MeanVE": "Mean minute ventilation",
  "second_half_VESlope": "Second half VE slope",
  "O2PulseDiff": "Actual - Expected maximum O2 pulse",
  "50_to_75_O2Slope": "Third quarter O2 pulse",
  "O2PulsePercent": "Actual/Expected maximum O2 pulse",
  "75_to_100_RERSlope": "Last quarter RER slope",
  "PeakRER": "Max RER",
  "50_to_75_VO2Slope": "Last quarter VO2 slope",
  "PeakVO2Real": "Max VO2"
}

pulmonary_feature_dict = {
  "O2PulsePercent": "Actual/Expected maximum O2 pulse",
  "O2PulseDiff": "Actual - Expected maximum O2 pulse",
  "first_half_VO2Slope": "First half VO2 slope",
  "LowestVE/VCO2": "Minimum VE/VCO2",
  "first_half_VCO2Slope": "First half VCO2 slope",
  "15_to_85_RRSlope": "15 to 85 session's percent of RR slope",
  "PeakRR": "Max RR",
  "50_to_75_RRSlope": "Third quarter RR slope",
  "MeanO2Pulse": "Mean O2 pulse",
  "VEvsVCO2Slope": "VE/VCO2 slope",
  "25_to_50_VCO2Slope": "Second quarter VCO2 slope",
  "StdHeartRate": "Heart rate's signal standard deviation"
}

other_feature_dict = {
  "PeakRR": "Max RR",
  "PeakVE": "Max minute ventilation",
  "PeakVCO2": "Max VCO2",
  "MeanVCO2": "Mean VCO2",
  "PeakVO2": "Max VO2",
  "PeakVO2Real": "Max VO2",
  "LowestVE/VCO2": "Minimum VE/VCO2",
  "MeanVCO2": "Mean VCO2",
  "O2PulsePercent": "Actual/Expected maximum O2 pulse",
  "O2PulseDiff": "Actual - Expected maximum O2 pulse",
  "first_half_VO2Slope": "First half VO2 slope",
  "LowestVE/VCO2": "Minimum VE/VCO2",
  "MeanRER": "Mean RER",
  "PeakRER": "Max RER",
  "VO2vsPeakVO2atVT": "VO2 at ventilatory threshold vs expected VO2",
  "DiffPercentPeakVO2": "Actual/Expected Peak VO2",
  "MeanRR": "Mean RR",
  "75_to_100_VEVCO2Slope": "Last quarter VE/VCO2 slope",
  "DiffPeakVO2": "Actual - Expected Peak VO2",
  "MeanVE": "Mean minute ventilation",
  "second_half_VESlope": "Second half VE slope",
  "first_half_VEVCO2Slope": "First half VE/VCO2 slope",
  "0_to_25_O2Slope": "First quarter O2 pulse slope",
  "VO2atVT": "VO2 at the moment of the ventilatory threshold",
  "MeanVO2": "Mean VO2",
  "second_half_VCO2Slope": "Second half VCO2 slope",
  "DiffPeakHR": "Actual-Expected peak heart rate",
  "MeanVE/VCO2": "Mean VE/VCO2 value",
  "75_to_100_RRSlope": "Last quarter RR slope"
}

def get_interpretation_images_by_id(session_id):
    try:
        data_df= pd.read_csv('./data/data_100.csv')
        session_id = float(session_id)
        image_list_str = []
        lim_types = ['cardiac', 'pulmonary', 'other']
        selected_model = None
        feature_selector = None
        df_data_renamed = None
        for lim_type in lim_types:
            if lim_type == 'cardiac':
                feature_selector = cardiac_data_100[1:]
                df_data_renamed = data_df[feature_selector]
                df_data_renamed.columns = df_data_renamed.columns.to_series().map(cardiac_feature_dict)
            elif lim_type == 'pulmonary':
                feature_selector = pulmonary_data_100[1:]
                df_data_renamed = data_df[feature_selector]
                df_data_renamed.columns = df_data_renamed.columns.to_series().map(pulmonary_feature_dict)
            else:
                feature_selector = other_data_100[1:]
                df_data_renamed = data_df[feature_selector]
                df_data_renamed.columns = df_data_renamed.columns.to_series().map(other_feature_dict)
            shap_values = pickle.load(
                    open(f"./models/{lim_type}/"+lim_type+'_shap_values.sav', 'rb'))
            data_index = data_df.loc[data_df['SessionId'] == session_id].index[0]
            plt.close()
            #pl_result = summary_with_highlight(shap_values[1], data_df[feature_selector], row_highlight=data_index, max_display=10, as_string=True)
            pl_result = summary_with_highlight(shap_values[1], df_data_renamed, row_highlight=data_index, max_display=10, as_string=True)
            image_list_str.append(pl_result)
            force_pl_str = create_force_plot_string(lim_type, data_index)
            image_list_str.append(force_pl_str)
        result = CPETInterpretationImages(image_list_str[0],image_list_str[1],image_list_str[2],
                                        image_list_str[3],image_list_str[4],image_list_str[5])
        return result, 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass


def create_force_plot_string(lim_type, data_index):
    file_name = str(uuid.uuid4())
    file_name_png = file_name + '.png'
    file_name_jpg = file_name + '.jpg'
    feature_selector = None
    renamed_feature_selector = None
    if lim_type == 'cardiac':
        feature_selector = cardiac_data_100[1:]
        renamed_feature_selector = [cardiac_feature_dict[elem] for elem in feature_selector]
    elif lim_type == 'pulmonary':
        feature_selector = pulmonary_data_100[1:]
        renamed_feature_selector = [pulmonary_feature_dict[elem] for elem in feature_selector]
    else:
        feature_selector = other_data_100[1:]
        renamed_feature_selector = [other_feature_dict[elem] for elem in feature_selector]
    loaded_tree = pickle.load(open(f"./models/{lim_type}/"+lim_type+'_tree_explainer.sav', 'rb'))
    shap_values = pickle.load(open(f"./models/{lim_type}/"+lim_type+'_shap_values.sav', 'rb'))

    
    import io
    my_stringIObytes = io.BytesIO()
    shap.force_plot(loaded_tree.expected_value[1], shap_values[1][data_index], feature_names=renamed_feature_selector,
            link='identity', contribution_threshold=0.1,show=False, plot_cmap=['#77dd77', '#f99191'],
            matplotlib=True).savefig(my_stringIObytes,format = "jpg",dpi = 150,bbox_inches = 'tight')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.getvalue()).decode("utf-8").replace("\n", "")
    pl.close()
    return str(my_base64_jpgData)


    shap.force_plot(loaded_tree.expected_value[1], shap_values[1][data_index], feature_names=renamed_feature_selector,
            link='identity', contribution_threshold=0.1,show=False, plot_cmap=['#77dd77', '#f99191'],
            matplotlib=True).savefig('./temp_images/'+file_name_png,format = "png",dpi = 150,bbox_inches = 'tight')
    png_img = img.open('./temp_images/'+file_name_png)
    rgb_im = png_img.convert('RGB')
    rgb_im.save('./temp_images/'+file_name_jpg,'JPEG')
    with open('./temp_images/'+file_name_jpg, 'rb') as image_file:
            b64_bytes = base64.b64encode(image_file.read())
    b64_string = str(b64_bytes, encoding='utf-8')
    plt.close()
    os.remove('./temp_images/'+file_name_png)
    os.remove('./temp_images/'+file_name_jpg)
    return b64_string
    pass

def _test_force_plot():
    #shap.initjs()
    f = pl.gcf()
    lim_type = 'cardiac'
    loaded_tree = pickle.load(open(f"./models/{lim_type}/"+lim_type+'_tree_explainer.sav', 'rb'))
    shap_values = pickle.load(open(f"./models/{lim_type}/"+lim_type+'_shap_values.sav', 'rb'))
    shap.force_plot(loaded_tree.expected_value[1], shap_values[-1][0], feature_names=cardiac_data_100[1:],
            link='identity', contribution_threshold=0.1,show=False, plot_cmap=['#77dd77', '#f99191'],
            matplotlib=True).savefig('./temp_images/'+'scratch.png',format = "png",dpi = 150,bbox_inches = 'tight')
    pass

def _save_tree_explainer_and_shaps():
    data_df= pd.read_csv('./data/data_100.csv')
    image_list_str = []
    lim_types = ['cardiac', 'pulmonary', 'other']
    selected_model = None
    selected_scaler = None
    feature_selector = None
    for lim_type in lim_types:
        if lim_type == 'cardiac':
            selected_model = pickle.load(
                open('./models/cardiac/clf_cardiac_100.sav', 'rb'))
            selected_scaler = pickle.load(
                open('./models/cardiac/scaler_clf_cardiac_100.sav', 'rb'))
            feature_selector = cardiac_data_100[1:]
        elif lim_type == 'pulmonary':
            selected_model = pickle.load(
                open('./models/pulmonary/clf_pulmonary_100.sav', 'rb'))
            selected_scaler = pickle.load(
                open('./models/pulmonary/scaler_clf_pulmonary_100.sav', 'rb'))
            feature_selector = pulmonary_data_100[1:]
        else:
            selected_model = pickle.load(
                open('./models/other/clf_other_100.sav', 'rb'))
            selected_scaler = pickle.load(
                open('./models/other/scaler_clf_other_100.sav', 'rb'))
            feature_selector = other_data_100[1:]
        selected_scaler.fit(data_df[feature_selector])
        X_scaled = selected_scaler.transform(data_df[feature_selector])
        explainer = shap.TreeExplainer(selected_model, data=X_scaled)
        shap_values = explainer.shap_values(X_scaled)
        print(explainer)
        print(shap_values)
        pickle.dump(explainer, open(f"./models/{lim_type}/"+lim_type+'_tree_explainer.sav','wb'))
        pickle.dump(shap_values, open(f"./models/{lim_type}/"+lim_type+'_shap_values.sav','wb'))
        loaded_explainer = pickle.load(
                    open(f"./models/{lim_type}/"+lim_type+'_tree_explainer.sav', 'rb'))
        loaded_shap_values = pickle.load(
                    open(f"./models/{lim_type}/"+lim_type+'_shap_values.sav', 'rb'))

        print(loaded_explainer)
        print(loaded_shap_values)
        print(str(type(loaded_shap_values)))
    pass

def get_cardiac_cpet_intepretation_by_id(session_id, lim_type):
    try:
        data_df= pd.read_csv('./data/data_100.csv')
        session_id = float(session_id)
        selected_model = None
        feature_selector = None
        if lim_type == 'cardiac':
            selected_model = pickle.load(
                open('./models/cardiac/clf_cardiac_100.sav', 'rb'))
            feature_selector = cardiac_data_100[1:]
        elif lim_type == 'pulmonary':
            selected_model = pickle.load(
                open('./models/pulmonary/clf_pulmonary_100.sav', 'rb'))
            feature_selector = pulmonary_data_100[1:]
        else:
            selected_model = pickle.load(
                open('./models/other/clf_other_100.sav', 'rb'))
            feature_selector = other_data_100[1:]
        explainer = shap.TreeExplainer(selected_model, data=data_df[feature_selector])
        shap_values = explainer.shap_values(data_df[feature_selector])
        data_index = data_df.loc[data_df['SessionId'] == session_id].index[0]
        pl_result = summary_with_highlight(shap_values[1], data_df[feature_selector], row_highlight=data_index, max_display=10, as_string=True)
        return pl_result, 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass

def get_dynamic_cpet_record_by_session_id(session_id):
    """API that retrieves the dynamic records of a patient

    This method validates the input and calls a method that will retrieve
    a list of all the health records of a patient, if there is one exists.
    The result will be a list of dictionaries containing the information

    Args:
        session_id (str): The id of the health record

    Returns
        `list` of `dict`, int: All patient's medical records, 200
        str, int: Error Message, 400
    """
    try:
        session_id = float(session_id)
        data_df = pd.read_csv('./data/data_export_dynamic.csv')
        data_filtered = data_df.loc[data_df.SessionId == session_id]
        cardiac_array = _generate_array(data_filtered, 'Cardiac')
        pulmonary_array = _generate_array(data_filtered, 'Pulmonary')
        other_array = _generate_array(data_filtered, 'Other')
        result = PatientDynamicFullPrediction(data_filtered.SessionId.values[0], data_filtered.PatientId.values[0], 
                                cardiac_array, data_filtered.CardiacLim.values[0],
                                pulmonary_array, data_filtered.PulmonaryLim.values[0],
                                other_array, data_filtered.OtherLim.values[0])
        return result, 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass

def get_cpet_record_by_session_id(session_id):
    """API that retrieves all the cpet reocrds of a session

    This method validates the input and calls a method that will retrieve
    a list of all the health records of a patient, if there is one exists.
    The result will be a list of dictionaries containing the information

    Args:
        session_id (str): The id of the health record

    Returns
        `list` of `dict`, int: All patient's medical records, 200
        str, int: Error Message, 400
    """
    try:
        session_id = float(session_id)
        data_df = pd.read_csv('./data/cpet_full_proba.csv')
        data_filtered = data_df.loc[data_df.SessionId == session_id]
        result = PatientFullPrediction(data_filtered.SessionId.values[0], data_filtered.PatientId.values[0], 
                                data_filtered.CardiacLimProba.values[0], data_filtered.CardiacLim.values[0],
                                data_filtered.PulmonaryProba.values[0], data_filtered.PulmonaryLim.values[0],
                                data_filtered.OtherProba.values[0], data_filtered.OtherLim.values[0])
        return result, 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass


if __name__ == "__main__":
    #result = [cardiac_feature_dict[elem] for elem in cardiac_data_100[1:]]
    #print(result)
    #_save_tree_explainer_and_shaps()
    get_cpet_record_by_session_id("7")
    #create_force_plot_string('cardiac', 7)
    #_test_force_plot()
    pass

