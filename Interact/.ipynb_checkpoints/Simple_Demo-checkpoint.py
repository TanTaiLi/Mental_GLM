import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(fn=greet, 
                    inputs=gr.Textbox(label="What's your name?"), 
                    outputs=gr.Text(label="Greeting"))

demo.launch(share=True)