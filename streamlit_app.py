import os
from pathlib import Path
from io import StringIO

import streamlit as st
import openai

import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st

from meeting_summarizer.fileloader import WebVttLoader, SrtLoader
from meeting_summarizer.prompter import SummarizerPrompter
from meeting_summarizer.config import AppConfig
from meeting_summarizer.summarizer import Summarizer
from meeting_summarizer.utils import LANGUAGES, TO_LANGUAGE_CODE
from meeting_summarizer.config import text_engine_choices
from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

st.title("Teams Meetings Summarize / Taskify")
st.write("Summarize.vtt/.srts files from your meeting transcripts")
transcript_name=st.file_uploader("Upload your .vtt or .srt files",type=['vtt','srt'])
col1,col2=st.columns(2)
openai_key=col1.text_input("OpenAI API Key",type="password")

text_engine_options = ["gpt-3.5-turbo","text-davinci-003"]
default_text_engine_option = "gpt-3.5-turbo"
text_engine_select=col2.selectbox("Text Engine",options=text_engine_options, index=text_engine_options.index(default_text_engine_option), help='Select the Open AI text engine for the summary')

languages_options = sorted(LANGUAGES.values())
default_lang_option = "english"
lang_select = col1.selectbox("Language",options=languages_options,index=languages_options.index(default_lang_option), help='Select the target language for the summary')

test=col2.checkbox("Test",value=True,help='Select this option to only summarize 4 contents you can easily check')

make_button=st.button("Make Transcript Summary")

st.session_state["summary"]=None

path="tmp"
if "summarize_sucess" not in st.session_state:
    st.session_state["summarize_sucess"] = False

if os.path.exists(path) == False:
    os.mkdir(path)

if transcript_name is not None:
    st.session_state["original_transcrpt_name"]=os.path.join(path,transcript_name.name)
    with open(st.session_state["original_transcrpt_name"],"wb") as f:
        f.write(transcript_name.getbuffer())

if make_button:
    text_engine = text_engine_choices.get(text_engine_select, "gpt-3.5-turbo")
    st.session_state["summarized_file_name"]=st.session_state["original_transcrpt_name"].split(".")[0]+".summary.txt"
    if os.path.exists(st.session_state["summarized_file_name"]) == False:
       
        streamlit_progress_bar = st.progress(0)
        streamlit_progress_message = st.markdown(" ")
        summarizing=st.markdown("Summarizing...")
        message = st.markdown(" ")

        file_path =st.session_state["original_transcrpt_name"]
        # Check the file is .vtt or .srt file
        file_extension = Path(file_path).suffix
        if file_extension not in [".vtt", ".srt"]:
            raise ValueError("File must be a .vtt /.srt file")
        # Initialize the loader class
        if file_extension == ".vtt":
            data_loader = WebVttLoader()
        elif file_extension == ".srt":
            data_loader = SrtLoader()
        # Initialize the config class
        config = AppConfig()
        config.set_text_engine(text_engine)
        config.LANGUAGE = lang_select
        if test:
            config.IS_TEST = True
            config.TEST_NUM = 4
        else:
            config.IS_TEST = False
        # Initialize the Summarizer class
        openai.api_key = openai_key
        summarizer = Summarizer(config,SummarizerPrompter, data_loader, streamlit_progress_bar=streamlit_progress_bar, streamlit_progress_message=streamlit_progress_message)
        summarizer.make_summary(file_path=st.session_state["original_transcrpt_name"], export_dir=path)
        streamlit_progress_bar.progress(100)
        streamlit_progress_message = st.markdown(" ")
        summarizing.markdown("Processing Done.")
        
    with open(st.session_state["summarized_file_name"],"rb") as f:
        st.session_state["summary"]=f.read()
    st.session_state["summarize_sucess"]=True

# delete the file
if st.session_state["summarize_sucess"]==True:
    try:
        os.remove(st.session_state["summarized_file_name"])
        os.remove(st.session_state["original_transcrpt_name"])

        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    except:
        pass 

if st.session_state["summarize_sucess"]==True and st.session_state["summary"] is not None:
    st.text_area(label ="",value=st.session_state["summary"], height =100)
    download_button=st.download_button(
        label="Download",
        data=st.session_state["summary"],
        file_name=transcript_name.name.split(".")[0]+".summary.txt",
    ) 

#################
### AUDIO ###
#################
    
# # This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
# def download_file(url, download_to: Path, expected_size=None):
#     # Don't download the file twice.
#     # (If possible, verify the download using the file length.)
#     if download_to.exists():
#         if expected_size:
#             if download_to.stat().st_size == expected_size:
#                 return
#         else:
#             st.info(f"{url} is already downloaded.")
#             if not st.button("Download again?"):
#                 return

#     download_to.parent.mkdir(parents=True, exist_ok=True)

#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading %s..." % url)
#         progress_bar = st.progress(0)
#         with open(download_to, "wb") as output_file:
#             with urllib.request.urlopen(url) as response:
#                 length = int(response.info()["Content-Length"])
#                 counter = 0.0
#                 MEGABYTES = 2.0 ** 20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)

#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning(
#                         "Downloading %s... (%6.2f/%6.2f MB)"
#                         % (url, counter / MEGABYTES, length / MEGABYTES)
#                     )
#                     progress_bar.progress(min(counter / length, 1.0))
#     # Finally, we remove these visual elements by calling .empty().
#     finally:
#         if weights_warning is not None:
#             weights_warning.empty()
#         if progress_bar is not None:
#             progress_bar.empty()


# def main():
#     st.sidebar.text('')
#     st.sidebar.text('')
#     st.sidebar.text('')
#     ### SEASON RANGE ###
#     #st.sidebar.markdown("**First select the data range you want to analyze:** 👇")
#     st.sidebar.header("Real Time Speech-to-Text")
#     st.sidebar.markdown(
#         """
#  Internallyusing [DeepSpeech](https://github.com/mozilla/DeepSpeech), an open speech-to-text engine.
# Using a pre-trained model [v0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3) trained on American English.
# """
#     )

#     # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
#     MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
#     LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
#     MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
#     LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

#     download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
#     download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

#     lm_alpha = 0.931289039105002
#     lm_beta = 1.1834137581510284
#     beam = 100

#     sound_only_page = "Sound only (sendonly)"
#     with_video_page = "With video (sendrecv)"
#     app_mode = st.sidebar.selectbox("Choose the app mode", [sound_only_page, with_video_page])

#     if app_mode == sound_only_page:
#         app_sst(
#             str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
#         )
#     elif app_mode == with_video_page:
#         app_sst_with_video(
#             str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
#         )


# def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=1024,
#         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#         media_stream_constraints={"video": False, "audio": True},
#     )

#     status_indicator = st.sidebar.empty()

#     if not webrtc_ctx.state.playing:
#         return

#     status_indicator.write("Loading...")
#     text_output = st.sidebar.empty()
#     stream = None

#     while True:
#         if webrtc_ctx.audio_receiver:
#             if stream is None:
#                 from deepspeech import Model

#                 model = Model(model_path)
#                 model.enableExternalScorer(lm_path)
#                 model.setScorerAlphaBeta(lm_alpha, lm_beta)
#                 model.setBeamWidth(beam)

#                 stream = model.createStream()

#                 status_indicator.write("Model loaded.")

#             sound_chunk = pydub.AudioSegment.empty()
#             try:
#                 audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             except queue.Empty:
#                 time.sleep(0.1)
#                 status_indicator.write("No frame arrived.")
#                 continue

#             status_indicator.write("Running. Say something!")

#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound

#             if len(sound_chunk) > 0:
#                 sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
#                     model.sampleRate()
#                 )
#                 buffer = np.array(sound_chunk.get_array_of_samples())
#                 stream.feedAudioContent(buffer)
#                 text = stream.intermediateDecode()
#                 text_output.markdown(f"**Text:** {text}")
#         else:
#             status_indicator.write("AudioReciver is not set. Abort.")
#             break


# def app_sst_with_video(
#     model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int
# ):
#     frames_deque_lock = threading.Lock()
#     frames_deque: deque = deque([])

#     async def queued_audio_frames_callback(
#         frames: List[av.AudioFrame],
#     ) -> av.AudioFrame:
#         with frames_deque_lock:
#             frames_deque.extend(frames)

#         # Return empty frames to be silent.
#         new_frames = []
#         for frame in frames:
#             input_array = frame.to_ndarray()
#             new_frame = av.AudioFrame.from_ndarray(
#                 np.zeros(input_array.shape, dtype=input_array.dtype),
#                 layout=frame.layout.name,
#             )
#             new_frame.sample_rate = frame.sample_rate
#             new_frames.append(new_frame)

#         return new_frames

#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text-w-video",
#         mode=WebRtcMode.SENDRECV,
#         queued_audio_frames_callback=queued_audio_frames_callback,
#         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#         media_stream_constraints={"video": True, "audio": True},
#     )

#     status_indicator = st.sidebar.empty()

#     if not webrtc_ctx.state.playing:
#         return

#     status_indicator.write("Loading...")
#     text_output = st.sidebar.empty()
#     stream = None

#     while True:
#         if webrtc_ctx.state.playing:
#             if stream is None:
#                 from deepspeech import Model

#                 model = Model(model_path)
#                 model.enableExternalScorer(lm_path)
#                 model.setScorerAlphaBeta(lm_alpha, lm_beta)
#                 model.setBeamWidth(beam)

#                 stream = model.createStream()

#                 status_indicator.write("Model loaded.")

#             sound_chunk = pydub.AudioSegment.empty()

#             audio_frames = []
#             with frames_deque_lock:
#                 while len(frames_deque) > 0:
#                     frame = frames_deque.popleft()
#                     audio_frames.append(frame)

#             if len(audio_frames) == 0:
#                 time.sleep(0.1)
#                 status_indicator.write("No frame arrived.")
#                 continue

#             status_indicator.write("Running. Say something!")

#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound

#             if len(sound_chunk) > 0:
#                 sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
#                     model.sampleRate()
#                 )
#                 buffer = np.array(sound_chunk.get_array_of_samples())
#                 stream.feedAudioContent(buffer)
#                 text = stream.intermediateDecode()
#                 text_output.markdown(f"**Text:** {text}")
#         else:
#             status_indicator.write("Stopped.")
#             break


# if __name__ == "__main__":
#     import os

#     DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

#     logging.basicConfig(
#         format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
#         "%(message)s",
#         force=True,
#     )

#     logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

#     st_webrtc_logger = logging.getLogger("streamlit_webrtc")
#     st_webrtc_logger.setLevel(logging.DEBUG)

#     fsevents_logger = logging.getLogger("fsevents")
#     fsevents_logger.setLevel(logging.WARNING)

#     main()
