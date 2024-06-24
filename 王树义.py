from fastai.tabular.all import*
import streamlit as st
import pandas as pd
from PIL import Image
from fastai.vision.core import PILImage
#导入模型
from joblib import load
import pickle
learn = load("E:\python(jupyter)\wwy\model.pkl")

# 加载蘑菇数据集
mashroom_df = pd.read_excel("E:\python(jupyter)\wwy\mashroom.xlsx")
mashroom_df = mashroom_df.rename_axis('mashroom_id').reset_index()

# 创建蘑菇ID字典
mashroom_id_dict = {
    'Amanita_Caesarea-Edible': 0,
    'Amanita_Citrina-Edible': 1,
    'Amanita_Pantherina-NotEdible': 2,
    'Boletus_Regius-Edible': 3,
    'Clitocybe_Costata-Edible': 4,
    'Entoloma_Lividum-NotEdible': 5,
    'Gyromitra_Esculenta-NotEdible': 6,
    'Helvella_Crispa-Edible': 7,
    'Hydnum_Rufescens-NotEdible': 8,
    'Hygrophorus_Latitabundus-Edible': 9,
    'Morchella_Deliciosa-Edible': 10,
    'Omphalotus_Olearius-NotEdible': 11,
    'Phallus_Impudicus-NotEdible': 12,
    'Rubroboletus_Satanas-NotEdible': 13,
    'Russula_Cyanoxantha-Edible': 14,
    'Russula_Delica-NotEdible': 15
}

# Streamlit应用开始
def main():
    st.title("蘑菇推荐系统")
    
    # 用户上传图片
    uploaded_file = st.file_uploader("上传蘑菇图片", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='上传的蘑菇图像', use_column_width=True)
        
        # 图像预测
        img = PILImage.create(image)
        pred, pred_idx, probs = learn.predict(img)
        st.write(f'预测类别: {pred}, 概率: {probs[pred_idx]:.2f}')
        
        # 获取用户上传蘑菇的ID
        user_mashroom_id = mashroom_id_dict.get(pred, -1)
        if user_mashroom_id != -1:
            # 显示预测的蘑菇鲜度和甜度
            row = mashroom_df.loc[user_mashroom_id]
            st.write(f"鲜度：{row['freshness']}，甜度：{row['sweetness']}")
            
            # 设置用户偏好输入
            st.header("设置您的偏好")
            user_freshness_min = st.slider("最低鲜度", min_value=1, max_value=10, value=row['freshness'])
            user_freshness_max = st.slider("最高鲜度", min_value=user_freshness_min, max_value=10, value=row['freshness']+1)
            user_sweetness_min = st.slider("最低甜度", min_value=1, max_value=5, value=row['sweetness'])
            user_sweetness_max = st.slider("最高甜度", min_value=user_sweetness_min, max_value=5, value=row['sweetness']+1)
            
            # 根据用户偏好推荐
            if st.button("推荐相似蘑菇"):
                recommended_mushrooms = recommend_mushrooms((user_freshness_min, user_freshness_max), (user_sweetness_min, user_sweetness_max))
                st.subheader("根据您的偏好推荐的蘑菇:")
                st.dataframe(recommended_mushrooms[['mashroom', 'freshness', 'sweetness']])
        else:
            st.error("无法识别的蘑菇种类，请确保上传清晰的图片。")

def recommend_mushrooms(user_freshness_range, user_sweetness_range):
    # 用户偏好
    user_freshness_min, user_freshness_max = user_freshness_range
    user_sweetness_min, user_sweetness_max = user_sweetness_range
    
    # 过滤符合用户偏好的蘑菇
    filtered_mushrooms = mashroom_df[(mashroom_df['freshness'] >= user_freshness_min) & 
                                    (mashroom_df['freshness'] <= user_freshness_max) & 
                                    (mashroom_df['sweetness'] >= user_sweetness_min) & 
                                    (mashroom_df['sweetness'] <= user_sweetness_max)]
    return filtered_mushrooms

if __name__ == "__main__":
    main()