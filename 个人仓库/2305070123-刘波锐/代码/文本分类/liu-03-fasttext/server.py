import fasttext
import jieba

from flask import Flask, request

app = Flask(__name__)


model = fasttext.load_model('fasttext_model2_jieba.bin')
@app.route('/v1/main_server/', methods=['POST'])
def main_server():
    text = request.form['text']
    # text = '广东省惠州学院装空调了'
    text = ' '.join(jieba.lcut(text))

    res = model.predict(text)
    res = res[0][0][9:]

    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)