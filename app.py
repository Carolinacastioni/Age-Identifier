import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import src.predictor as pre

modelo_json = 'Model_0.7313541173934937_18-22-26.h5.json'
modelo_h5 = 'Model_0.7313541173934937_18-22-26.h5'


# instancia del objeto Flask
app = Flask(__name__)

# Carpeta de subida
app.config['UPLOAD_FOLDER'] = 'result'

 # renderiamos la plantilla "index.html"
@app.route("/")
def landing_page():
    return render_template('index.html')


@app.route("/upload", methods=['POST'])
def uploader():
  if request.method == 'POST':
    # obtenemos el archivo del input "archivo"
    f = request.files['archivo']
    filename = secure_filename(f.filename)
    # Guardamos el archivo en el directorio "result"
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # Aplicamos el modelo en la imagen"
    img = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pred = pre.agePrediction(img,modelo_json, modelo_h5)
    return render_template("predictor.html", pred=pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3500, debug=True)