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
    st.file_uploader(
        label='stock_pdf', label_visibility='collapsed', type='pdf'
    )

st.divider()

tabs_name = ('Rangkuman', 'Potensi', 'Risiko')
tabs = st.tabs(tabs_name)

with tabs[0]:
    st.write(
        '''
    Saham PT Indosat Tbk (ISAT) anjlok tajam usai merilis laporan keuangan 2024 pada Senin (10/2), dengan harga turun 13,06 persen dari pembukaan di 2.220 ke 1.930 per lembar menjelang penutupan perdagangan. Penurunan signifikan terjadi setelah sesi I, ketika harga masih di 2.200 lalu melorot agresif hingga 1.965 dan akhirnya 1.930. Kondisi ini memicu diskusi hangat di forum Stockbit, di mana investor terbagi antara bertahan, cut loss, atau membeli saat harga turun meski laba tahunan ISAT meningkat.
'''
    )