# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
from text_classification import TEXT_CLASSIFICATION

def create_app():
    app = Flask(__name__)
    tc = TEXT_CLASSIFICATION()

    @app.route('/lyric_classification',methods=['GET','POST'])
    def lyric_classification():
        if request.method == 'POST':
            data= request.form.to_dict()
            text_list = data['datas'].split(';')

            json_list = []
            for text in text_list:
                result = tc.pipe(text)
                result[0]['label'] = tc.label_dic[result[0]['label']]
                result[0]['text'] = text

                json_list.append(result[0])

            return_json = {"results":json_list}
            return jsonify(return_json)

    return app


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(host='0.0.0.0', port=5002)
