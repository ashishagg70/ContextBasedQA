from flask import Blueprint
from flask import request
import json
import numpy as np
from .qna_processing import predict_answer

bp = Blueprint('qna', __name__)

@bp.route('/', methods=['POST'])
def rating():
    req = request.json
    if req['context'] is None or len(req['context'])==0:
        return {
            "status": 403,
            "message": "Please send context"
        }
    if req['question'] is None or len(req['question'])==0:
        return {
            "status": 403,
            "message": "Please send question"
        }

    answer = predict_answer(req['context'],req['question'])
    return {
        "status":200,
        "context":req['context'],
        "question":req['question'],
        "answer":answer
    }

    