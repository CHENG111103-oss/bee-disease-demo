import streamlit as st
from PIL import Image
from ultralytics import YOLO

model = YOLO(r"D:/bee_project/runs/classify/train/weights/best.pt")

st.title("🐝 蜜蜂病害智能识别系统（原型）")
st.write("上传一张蜜蜂或蜂巢图片，系统将自动识别病害类型")

uploaded_file = st.file_uploader("请选择一张图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="输入图片", use_column_width=True)

    if st.button("开始识别"):
        results = model(image)

        probs = results[0].probs
        class_id = probs.top1
        confidence = probs.top1conf.item()
        class_name = model.names[class_id]

        st.success(f"识别结果：**{class_name}**")
        st.write(f"置信度：{confidence:.2%}")
