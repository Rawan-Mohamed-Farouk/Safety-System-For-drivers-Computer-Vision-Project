import streamlit as st
from face_verification import face_verification
from driver_monitoring import driver_monitoring

st.title("Driver Authentication & Monitoring System")

if st.button("Start System"):
    st.write("🔍 Starting Face Verification...")
    if face_verification():
        st.success("✅ Welcome, Rawan")
        st.write("🚗 Starting Driver Monitoring System...")
        driver_monitoring()
    else:
        st.error("❌ Access Denied! Unknown Person.")
        
        #cli: python -m streamlit run "C:\Users\User\Desktop\BS\finalll\app.py"