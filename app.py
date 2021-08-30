from flask import Flask, jsonify, Request
from flask_cors import CORS
import os
import current_patient_service as cps
import json

app = Flask(__name__)
CORS(app)


@app.route("/api/get_shap_interpretation_by_session_id/<session_id>",
            methods=["GET"])
def get_cardiac_cpet_intepretation_by_id(session_id):
    """API that returns a base64 of the force plot and custom shap summary

    This method validates the input and calls a method that will process a
    list of values that represents the time and voltage of an ecg signal.
    It plots the information and enconde it as a base64 string and without
    storing it into the database.

    Args:
        in_data (str): session id

    Returns
        str,str, int: SHAP and force plot image as a base64 str, 200
        str, int: Error Message, 400
    """
    try:
        shap_custom_result = cps.get_interpretation_images_by_id(session_id)
        #res, code = hr.get_record_by_patient_id(patient_id)
        return json.dumps(vars(shap_custom_result[0])), 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass


@app.route("/api/get_dynamic_record_by_session_id/<session_id>",
            methods=["GET"])
def get_dynamic_cpet_by_session_id(session_id):
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
        result = cps.get_dynamic_cpet_record_by_session_id(session_id)
        #res, code = hr.get_record_by_patient_id(patient_id)
        return json.dumps(vars(result[0])), 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400
    pass

@app.route("/api/get_record_by_patient_id/<session_id>",
           methods=["GET"])
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
        result = cps.get_cpet_record_by_session_id(session_id)
        return json.dumps(vars(result[0])), 200
    except Exception as e:
        print(e)
        return "Unexpected error", 400


@app.route('/')
def root():
    return "Hello World!"

if __name__ == "__main__":
        # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
    #app.run(host="0.0.0.0", port=5000)