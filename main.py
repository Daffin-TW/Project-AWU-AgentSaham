from streamlit import secrets as ss
import streamlit as st


st.markdown('### Agent AI Laporan Keuangan Persusahaan')

st.divider()

col1, col2 = st.columns((1, 2))

with col1.container(border=False):
    url1 = (
        'https://www.idx.co.id/id/perusahaan-tercatat/' +
        'laporan-keuangan-dan-tahunan/'
    )
    url2 = (
        'https://www.idx.co.id/id/perusahaan-tercatat/' +
        'profil-perusahaan-tercatat/'
    )

    st.markdown('#### Sumber file laporan keuangan')

    bcol1, bcol2 = st.columns(2)
    bcol1.link_button('Link 1', url=url1, use_container_width=True)
    bcol2.link_button('Link 2', url=url2, use_container_width=True)

with col2.container(border=False):
    st.markdown('#### Unggah file laporan keuangan')
    st.file_uploader(label='stock_pdf', label_visibility='collapsed')

st.divider()

tabs = ('Rangkuman', 'Potensi', 'Risiko')
st.tabs(tabs)