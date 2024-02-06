import base64
import json
import queue
import sys
import threading
import time

import pandas as pd
import plotly
import streamlit as st
import websocket
from audio_recorder_streamlit import audio_recorder
from streamlit.runtime import Runtime
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.web import cli as stcli

from besser.bot.core.file import File
from besser.bot.platforms.payload import Payload, PayloadAction, PayloadEncoder
from besser.bot.platforms.websocket.message import Message

# Time interval to check if a streamlit session is still active, in seconds
SESSION_MONITORING_INTERVAL = 10


def get_streamlit_session() -> AppSession or None:
    session_id = get_script_run_ctx().session_id
    runtime: Runtime = Runtime.instance()
    return next((
        s.session
        for s in runtime._session_mgr.list_sessions()
        if s.session.id == session_id
    ), None)


def session_monitoring(interval: int):
    runtime: Runtime = Runtime.instance()
    session = get_streamlit_session()
    while True:
        time.sleep(interval)
        if not runtime.is_active_session(session.id):
            runtime.close_session(session.id)
            session.session_state['websocket'].close()
            break


def main():
    st.set_page_config(
        page_title="Streamlit Chat - Demo",
        page_icon=":robot:"
    )
    st.header("Chat Demo")
    st.markdown("[Github](https://github.com/BESSER-PEARL/BESSER-Bot-Framework)")
    # User input component. Must be declared before history writing
    user_input = st.chat_input("What is up?")

    def on_message(ws, payload_str):
        # https://github.com/streamlit/streamlit/issues/2838
        streamlit_session = get_streamlit_session()
        payload: Payload = Payload.decode(payload_str)
        if payload.action == PayloadAction.BOT_REPLY_STR.value:
            content = payload.message
            t = 'str'
        elif payload.action == PayloadAction.BOT_REPLY_FILE.value:
            content = payload.message
            t = 'file'
        elif payload.action == PayloadAction.BOT_REPLY_DF.value:
            content = pd.read_json(payload.message)
            t = 'dataframe'
        elif payload.action == PayloadAction.BOT_REPLY_PLOTLY.value:
            content = plotly.io.from_json(payload.message)
            t = 'plotly'
        elif payload.action == PayloadAction.BOT_REPLY_OPTIONS.value:
            t = 'options'
            d = json.loads(payload.message)
            content = []
            for button in d.values():
                content.append(button)
        message = Message(t, content, is_user=False)
        streamlit_session._session_state['queue'].put(message)
        streamlit_session._handle_rerun_script_request()

    def on_error(ws, error):
        pass

    def on_open(ws):
        pass

    def on_close(ws, close_status_code, close_msg):
        pass

    def on_ping(ws, data):
        pass

    def on_pong(ws, data):
        pass

    user_type = {
        0: 'assistant',
        1: 'user'
    }

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'queue' not in st.session_state:
        st.session_state['queue'] = queue.Queue()

    if 'websocket' not in st.session_state:
        ws = websocket.WebSocketApp("ws://localhost:8765/",
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close,
                                    on_ping=on_ping,
                                    on_pong=on_pong)
        websocket_thread = threading.Thread(target=ws.run_forever)
        add_script_run_ctx(websocket_thread)
        websocket_thread.start()
        st.session_state['websocket'] = ws

    if 'session_monitoring' not in st.session_state:
        session_monitoring_thread = threading.Thread(target=session_monitoring,
                                                     kwargs={'interval': SESSION_MONITORING_INTERVAL})
        add_script_run_ctx(session_monitoring_thread)
        session_monitoring_thread.start()
        st.session_state['session_monitoring'] = session_monitoring_thread

    ws = st.session_state['websocket']

    with st.sidebar:

        if reset_button := st.button(label="Reset bot"):
            st.session_state['history'] = []
            st.session_state['queue'] = queue.Queue()
            payload = Payload(action=PayloadAction.RESET)
            ws.send(json.dumps(payload, cls=PayloadEncoder))

        if voice_bytes := audio_recorder(text=None, pause_threshold=2):
            if 'last_voice_message' not in st.session_state or st.session_state['last_voice_message'] != voice_bytes:
                st.session_state['last_voice_message'] = voice_bytes
                # Encode the audio bytes to a base64 string
                voice_message = Message(t='audio', content=voice_bytes, is_user=True)
                st.session_state.history.append(voice_message)
                voice_base64 = base64.b64encode(voice_bytes).decode('utf-8')
                payload = Payload(action=PayloadAction.USER_VOICE, message=voice_base64)
                try:
                    ws.send(json.dumps(payload, cls=PayloadEncoder))
                except Exception as e:
                    st.error('Your message could not be sent. The connection is already closed')
        if uploaded_file := st.file_uploader("Choose a file", accept_multiple_files=False):
            if 'last_file' not in st.session_state or st.session_state['last_file'] != uploaded_file:
                st.session_state['last_file'] = uploaded_file
                bytes_data = uploaded_file.read()
                file_object = File(file_base64=base64.b64encode(bytes_data).decode('utf-8'), file_name=uploaded_file.name, file_type=uploaded_file.type)
                payload = Payload(action=PayloadAction.USER_FILE, message=file_object.get_json_string())
                file_message = Message(t='file', content=file_object.to_dict(), is_user=True)
                st.session_state.history.append(file_message)
                try:
                    ws.send(json.dumps(payload, cls=PayloadEncoder))
                except Exception as e:
                    st.error('Your message could not be sent. The connection is already closed')
    for message in st.session_state['history']:
        with st.chat_message(user_type[message.is_user]):
            if message.type == 'audio':
                st.audio(message.content, format="audio/wav")
            elif message.type == 'file':
                file: File = File.from_dict(message.content)
                file_name = file.name
                file_type = file.type
                file_data = base64.b64decode(file.base64.encode('utf-8'))
                st.download_button(label='Download ' + file_name, file_name=file_name, data=file_data, mime=file_type,
                                   key=file_name + str(time.time()))
            else:
                st.write(message.content)

    first_message = True
    while not st.session_state['queue'].empty():
        message = st.session_state['queue'].get()
        if hasattr(message, '__len__'):
            t = len(message.content) / 1000 * 3
        else:
            t = 2
        if t > 3:
            t = 3
        elif t < 1 and first_message:
            t = 1
        first_message = False
        if message.type == 'options':
            st.session_state['buttons'] = message.content
        elif message.type == 'file':
            st.session_state['history'].append(message)
            with st.chat_message('assistant'):
                with st.spinner(''):
                    time.sleep(t)
                file: File = File.from_dict(message.content)
                file_name = file.name
                file_type = file.type
                file_data = base64.b64decode(file.base64.encode('utf-8'))
                st.download_button(label='Download ' + file_name, file_name=file_name, data=file_data, mime=file_type,
                                   key=file_name + str(time.time()))
        else:
            st.session_state['history'].append(message)
            with st.chat_message("assistant"):
                with st.spinner(''):
                    time.sleep(t)
                st.write(message.content)

    if 'buttons' in st.session_state:
        buttons = st.session_state['buttons']
        cols = st.columns(1)
        for i, option in enumerate(buttons):
            if cols[0].button(option):
                with st.chat_message("user"):
                    st.write(option)
                message = Message(t='str', content=option, is_user=True)
                st.session_state.history.append(message)
                payload = Payload(action=PayloadAction.USER_MESSAGE,
                                  message=option)
                ws.send(json.dumps(payload, cls=PayloadEncoder))
                del st.session_state['buttons']
                break

    if user_input:
        if 'buttons' in st.session_state:
            del st.session_state['buttons']
        with st.chat_message("user"):
            st.write(user_input)
        message = Message(t='str', content=user_input, is_user=True)
        st.session_state.history.append(message)
        payload = Payload(action=PayloadAction.USER_MESSAGE,
                          message=user_input)
        try:
            ws.send(json.dumps(payload, cls=PayloadEncoder))
        except Exception as e:
            st.error('Your message could not be sent. The connection is already closed')

    st.stop()


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
