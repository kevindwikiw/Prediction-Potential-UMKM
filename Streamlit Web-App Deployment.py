import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))

def predict_label(Rating,Total_Reviews,Review_Relevan, SME_Data,Distance,Check_Web,Check_Telepon):
    input=np.array([[Rating, Total_Reviews, Review_Relevan, SME_Data, Distance, Check_Web, Check_Telepon]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{2}f}'.format(prediction[0][0], 2)
    return float(pred)
	
def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Classification Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    Rating = st.text_input("Rating","Type Here")
    Total_Reviews = st.text_input("Total_Reviews","Type Here")
    Review_Relevan = st.tex_input("Review_Relevan","Type Here")
    SME_Data = st.text_input("SME_Data","Type Here")
    Distance = st.text_input("Distance","Type Here")
    Check_Web = st.text_input("Check_Web","Type Here")
    Check_Telepon = st.text_input("Check_Telepon","Type Here")
    L2_html="""
    <div style="background-color:#F4D03F;padding:10px >
    <h2 style="color:white;text-align:center;"> Label Sangat Potential</h2>
    </div>
    """
    L1_html="""  
    <div style="background-color:#F4D03F;padding:10px >
    <h2 style="color:white ;text-align:center;"> Label Potential</h2>
    </div>
    """
    L0_html="""
    <div style="background-color:#F08080;padding:10px >
    <h2 style="color:black ;text-align:center;"> Label Tidak Potential</h2>
    </div>
    """
    
    if st.button("Predict"):
        output=predict_label(Rating,Total_Reviews,Review_Relevan,SME_Data,Distance,Check_Web,Check_Telepon)
        st.success('Prediksi klasifikasi label adalah {}'.format(output))
        if output == 0:
            st.markdown(L0_html,unsafe_allow_html=True)
        elif output == 1:
            st.markdown(L1_html,unsafe_allow_html=True)
        else:
            st.markdown(L2_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()