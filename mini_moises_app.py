import streamlit as st
import os
import torchaudio
import librosa
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
import tempfile
import numpy as np

st.set_page_config(page_title="Mini Moises 🎵", layout="wide")
st.title("🎧 Mini Moises - Separação de faixas")

uploaded_file = st.file_uploader("Faça upload de uma música (.mp3 ou .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        st.info("Preparando o arquivo...")

        input_path = os.path.join(tmpdir, uploaded_file.name)
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        try:
            st.info("Carregando o arquivo de áudio usando Librosa...")
            # Carrega com Librosa, MANTENDO os canais originais (mono=False)
            wav_np, sr = librosa.load(input_path, sr=None, mono=False)
            st.info(f"Áudio carregado. Taxa original: {sr} Hz. Shape original: {wav_np.shape}")

            # Verifica se é mono (shape [N,]) ou estéreo (shape [2, N])
            if wav_np.ndim == 1:
                st.info("Áudio é mono. Duplicando canal para formato estéreo.")
                # Duplica o canal mono para criar um estéreo falso [2, N]
                wav_np_stereo = np.stack([wav_np, wav_np])
            elif wav_np.shape[0] == 2:
                st.info("Áudio já é estéreo.")
                wav_np_stereo = wav_np
            else:
                st.error(f"Formato de canais inesperado: {wav_np.shape}. Processando apenas os 2 primeiros.")
                wav_np_stereo = wav_np[:2, :]

            # Converte para Tensor PyTorch [2, N]
            wav_stereo_tensor = torch.from_numpy(wav_np_stereo)

            st.info("Carregando o modelo Demucs...")
            model = get_model(name="htdemucs")
            model_sr = model.samplerate
            st.info(f"Modelo Demucs carregado. Taxa esperada: {model_sr} Hz")

            # Resampleia se necessário
            if sr != model_sr:
                st.info(f"Resampling áudio de {sr} Hz para {model_sr} Hz...")
                # IMPORTANTE: Resample espera [..., N], então passamos o tensor estéreo
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
                wav_resampled = resampler(wav_stereo_tensor)
                st.info("Resampling concluído.")
            else:
                wav_resampled = wav_stereo_tensor # Já está na taxa correta

            # Adiciona a dimensão do BATCH -> [1, 2, N] (batch, canal, tempo)
            wav_batch = wav_resampled.unsqueeze(0)
            st.info(f"Formato do tensor final para Demucs: {wav_batch.shape}")

            # Separa as faixas
            st.info("Separando as faixas (pode levar um tempo)...")
            # Passa o tensor com a dimensão de batch e 2 canais
            # O resultado 'sources' é um tensor com as faixas separadas, na ordem do modelo
            separated_sources = apply_model(model, wav_batch, split=True, progress=True)[0]

            st.success("Separação concluída!")

            st.subheader("🎚️ Faixas separadas:")
            # Nomes das faixas no modelo htdemucs
            source_names = ['drums', 'bass', 'other', 'vocals']

            for i, name in enumerate(source_names):
                audio_tensor = separated_sources[i]
                out_path = os.path.join(tmpdir, f"{name}.wav")
                # Salva o tensor estéreo usando a taxa de amostragem DO MODELO
                torchaudio.save(out_path, audio_tensor, model_sr)
                st.write(f"**{name.capitalize()}**")
                st.audio(out_path, format="audio/wav")

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento:")
            st.exception(e)
            st.error("Verifique o arquivo e as instalações. Tente reiniciar.")
#moises 