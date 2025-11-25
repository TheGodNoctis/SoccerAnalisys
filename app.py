import streamlit as st
import tempfile
from pathlib import Path
import os

st.set_page_config(
    page_title="Análise Tática de Futebol",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None  
)

PALETTE = {
    "primary": "#2ECC71",   # verde principal
    "accent":  "#45D98E",   # verde claro de destaque
    "bg":      "#F4F8F4",   # fundo claro
    "card":    "#2ECC71",   # fundo dos cards
    "muted":   "#5A5E63",   # texto sutil
}
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}   /* Esconde o menu hamburger */
    header {visibility: hidden;}      /* Esconde o header/deploy */
    footer {visibility: hidden;}      /* Esconde o rodapé padrão */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    
    f"""
    <style>
        
        /* Centraliza e limita largura do conteúdo */
        .block-container {{
            max-width: 1200px;
            padding-top: 3.5rem;
            padding-bottom: 3rem;
        }}

        /* Fundo geral */
        .stApp {{
            background:
                radial-gradient(1200px 800px at 10% 10%, rgba(46,204,113,0.15), transparent 40%),
                radial-gradient(1200px 800px at 90% 20%, rgba(69,217,142,0.12), transparent 40%),
                {PALETTE["bg"]};
            color: #1A1D21;
        }}

        /* Header */
        .hero {{
            border-radius: 20px;
            padding: 24px 24px 20px 24px;
            background: linear-gradient(135deg, {PALETTE["primary"]}22, {PALETTE["accent"]}22);
            border: 1px solid #D6D9DD;
            box-shadow: 0 5px 18px rgba(0,0,0,0.08);
        }}
        .hero h1 {{
            margin: 0 0 8px 0 !important;
            font-size: 2rem;
            letter-spacing: .2px;
            color: #1A1D21;
        }}
        .hero p {{
            margin: 0;
            color: {PALETTE["muted"]};
        }}

        /* Cards */
        .card {{
            background: {PALETTE["card"]};
            border: 1px solid #E1E4E8;
            border-radius: 16px;
            padding: 16px 16px 12px 16px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        }}
        .card h4 {{
            margin: 2px 0 10px 0;
            font-size: 1.05rem;
            color: #1A1D21;
        }}

        /* Botões padrão */
        .stButton>button {{
            background: linear-gradient(135deg, {PALETTE["primary"]}, {PALETTE["accent"]});
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            font-weight: 700;
            letter-spacing: .2px;
            box-shadow: 0 5px 14px rgba(46,204,113,0.35);
        }}
        .stButton>button:hover {{
            filter: brightness(1.05);
        }}

        /* Inputs */
        .stFileUploader, .stNumberInput, .stCheckbox, .stSelectbox, .stTextInput {{
            background: transparent !important;
            color: #1A1D21 !important;
        }}

        /* Texto suave */
        .muted {{
            color: {PALETTE["muted"]};
            font-size: 0.92rem;
        }}

        /* Linha separadora */
        .divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, #D8DBDF, transparent);
            margin: 8px 0 0 0;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.markdown("### ⚙️ Opções do Sistema")
    use_stubs = st.checkbox("Stubs", value=False, key="use_stubs")
    fps = st.number_input("🎞️ FPS do vídeo", min_value=1, max_value=120, value=24, key="fps")

    st.markdown("---")
    st.markdown("### 📋 Instruções de Uso")
    st.markdown(
        """
        1. **Faça o upload** do vídeo de uma partida (formato MP4 recomendado).  
        2. O sistema processará automaticamente o vídeo: detecção, rastreamento, passes e posse de bola.  
        3. Após o processamento, **serão exibidos botões para visualização** do vídeo anotado e do relatório.  
        4. Esses botões **ficam disponíveis até que um novo vídeo seja enviado**.  
        5. É possível ajustar o **FPS** caso o vídeo seja diferente de 24 FPS padrão.  
        6. Use a opção *stubs* apenas durante o desenvolvimento, para acelerar os testes.  
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 💡 Dicas Importantes")
    st.markdown(
        """
        - Use vídeos com boa **resolução e estabilidade** da câmera para resultados mais precisos.  
        - Certifique-se de que **boa parte do campo** esteja visível durante o jogo.  
        - O **relatório PDF** traz estatísticas detalhadas de posse, passes e movimentação.  
        - Recomenda-se vídeos **com pelo menos 10s**.
        - Vídeos com altas resoluções e tamanho tendem a levar longos tempos de processamento.
        """
    )

col_left, col_right = st.columns([1.05, 1])

import uuid, shutil  

def _reset_for_new_upload(uploaded_file):
    old_path = st.session_state.get("tmp_input_path")
    if old_path:
        try:
            shutil.rmtree(Path(old_path).parent, ignore_errors=True)
        except Exception:
            pass

    tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
    input_path = tmp_dir / uploaded_file.name
    input_path.write_bytes(uploaded_file.read())

    st.session_state.tmp_input_path = str(input_path)
    st.session_state.outputs = None
    st.session_state.job_ready = False
    st.session_state.last_upload_sig = f"{uploaded_file.name}:{uploaded_file.size}"
    st.success(f"Arquivo recebido: **{uploaded_file.name}**")

col_left, col_right = st.columns([1.05, 1])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📤 Upload do vídeo")
    uploaded = st.file_uploader(
        "Arraste e solte ou selecione um arquivo (mp4/avi/mov/mkv)",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        sig = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("last_upload_sig") != sig:
            _reset_for_new_upload(uploaded)

    st.write("")
    st.session_state.setdefault("tmp_input_path", None)
    process_enabled = st.session_state.tmp_input_path is not None
    clicked = st.button("▶️ Processar vídeo", disabled=not process_enabled, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 👀 Pré-visualização")

    if st.session_state.tmp_input_path:
        path = Path(st.session_state.tmp_input_path)
        try:
            with open(path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
        except Exception as e:
            st.warning(f"Falha ao abrir bytes do vídeo ({e}). Tentando via caminho.")
            st.video(str(path))
            st.caption(f"Arquivo temporário: `{path.name}`")
    else:
        st.info("Envie um vídeo para visualizar aqui.")

    st.markdown("</div>", unsafe_allow_html=True)

if clicked:
    with st.spinner("Processando o vídeo… isso pode levar alguns minutos."):
        try:
            from pipeline import process_video
            outputs = process_video(
                st.session_state.tmp_input_path,
                use_stubs=st.session_state.use_stubs,
                fps=int(st.session_state.fps),
                ann={
                    "mini_pitch": True,        
                    "speed_distance": False,   
                    "possession": False,      
                },
)
        except Exception as e:
            st.session_state.outputs = None
            st.session_state.job_ready = False
            st.error(f"Erro ao processar: {e}")
        else:
            st.session_state.outputs = outputs
            st.session_state.job_ready = True
            st.success("✅ Processamento concluído!")

st.session_state.setdefault("tmp_input_path", None)
st.session_state.setdefault("job_ready", False)
st.session_state.setdefault("outputs", None)

st.write("")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### Resultados")

if st.session_state.job_ready and st.session_state.outputs:
    outputs = st.session_state.outputs
    rep_path = Path(outputs["report_path"])
    vid_path = Path(outputs["video_path"])
    process_dir = Path(outputs.get("process_dir", ""))  

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("##### Relatório")
        if rep_path.exists():
            if st.button("📄 Abrir relatório", use_container_width=True):
                os.startfile(str(rep_path))
        else:
            st.warning("Relatório não encontrado.")

    with c2:
        st.markdown("##### Vídeo")
        if vid_path.exists():
            if st.button("🎬 Abrir vídeo processado", use_container_width=True):
                os.startfile(str(vid_path))
        else:
            st.warning("Vídeo processado não encontrado.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.write("")
    if st.button("🔄 Novo upload", use_container_width=True):
        st.session_state.outputs = None
        st.session_state.job_ready = False
        st.session_state.tmp_input_path = None
        st.session_state.pop("video_uploader", None)
        st.rerun()
    else:
        st.info("Envie um arquivo e clique em **Processar vídeo**.")
    st.markdown("</div>", unsafe_allow_html=True)
