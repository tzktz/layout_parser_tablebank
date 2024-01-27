import os
os.system('pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"')
import layoutparser as lp
import gradio as gr

# Initialize the Table Bank model for table detection
model_table_bank = lp.Detectron2LayoutModel('lp://TableBank/faster_rcnn_R_101_FPN_3x/config')


# Define the layout analysis function
def detect_tables(img):
    tables_layout = model_table_bank.detect(img)
    img_with_tables = lp.draw_box(img, tables_layout)
    return img_with_tables


# Set up Gradio interface
inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = [gr.outputs.Image(type="pil", label="Output Image with Tables")]

title = "Table Detection with Layout Parser (Table Bank Model)"
css = ".output-image, .input-image, .image-preview {height: 600px !important}"

article = "<p style='text-align: center'><a href='https://github.com/Layout-Parser/layout-parser'>Layout Parser GitHub Repo</a></p>"

gr.Interface(detect_tables, inputs, outputs, title=title, article=article, css=css).launch()
