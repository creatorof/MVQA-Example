import streamlit as st
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

st.title("Visual Question Answering System")

st.sidebar.header("Ask a question about the image")
image_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
question = st.sidebar.text_input("Your question")

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if st.sidebar.button("Get Answer") and image_file is not None and question:
    with st.spinner("Getting the answer..."):
        encoding = processor(image, question, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        st.success(f"Answer: {answer}")

st.info(
    "Instructions: Upload an image and enter your question in the sidebar. "
    "Then click 'Get Answer' to receive a response."
)
