Keshav Agarwal (203050039)
Ashish Aggarwal (203050015)
Debabrata Biswal (203050024)


Steps to install and run:
Download the model state_dict from drive and change the filepath in ContextBasedQnA/ContextBasedQnAAPI/qna_processing
https://drive.google.com/drive/folders/1ERLTh5GRCUtJqyWst17yxLvUDox3gA_G

Flask Server:
1. Create Python virtual environment and activate it:
    a.  python3 -m venv venv
    b.  Activate for windows: 
            venv\Scripts\activate
        Activate for Linux:
            . venv/bin/activate

2. Install packages 
    a. pip install flask
    b. pip install requests
    c. pip install flask-cors 
    d. pip install pandas
    e. pip install torch
    f. pip install nltk
3. cd ContextBasedQnA
4. For PowerShell, run:
       $env:FLASK_APP="ContextBasedQnAAPI"
   For Linux, run:
       export FLASK_APP="ContextBasedQnAAPI"
5. For PowerShell, run:
       $env:FLASK_ENV="developoment"
   For Linux, run:
       export FLASK_ENV="developoment"

6. flask run

Frontend:
7. Open following webpage in browser: ContextBasedQnA/ContextBasedQnAUI/index.html
