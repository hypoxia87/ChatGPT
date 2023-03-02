import datetime
import openai
import os
import pyaudio
import queue
import re
import threading
import wave
import whisper

API_KEY_PATH = os.path.expanduser('~/.openai')
AUDIO_QUERIES = True

MODEL = 'gpt-3.5-turbo'
KNOWLEDGE_CUTOFF = '2021-09-01'
CURRENT_DATE = datetime.date.today().isoformat()
CONTEXT = [] # Rolling context for subsequent interactions
HISTORY = [] # History of inputs/outputs

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_FILE = 'temp_output.wav'

# Threading shared variables
STOP_FLAG = threading.Event()
SHARED_QUEUE = queue.Queue()

### Load API key and test authentication ###

print('Loading OpenAI API key... ', end='')
try:
    with open(API_KEY_PATH, 'r') as fi:
        API_KEY = str(fi.read(1024)).strip()
    assert API_KEY
    print('Done.')
except Exception as e:
    print('Failed.')
    print('Please save an OpenAI API key to ~/.openai .')
    raise e from None

print('Authenticating... ', end='')
openai.api_key = API_KEY
try:
    openai.Model.list()
    print('Done.')
except Exception as e:
    print('Failed.')
    print('Please update your OpenAI API key in ~/.openai .')
    raise e from None

### Audio recording helper functions ###

def record_audio():
    global STOP_FLAG
    global SHARED_QUEUE
    p = pyaudio.PyAudio()
    # Open a stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if STOP_FLAG.is_set():
            break
    stream.stop_stream()
    stream.close()
    p.terminate()
    SHARED_QUEUE.put(frames)

def get_audio_query():
    global STOP_FLAG
    global SHARED_QUEUE
    STOP_FLAG = threading.Event()
    SHARED_QUEUE = queue.Queue()
    print('[Recording. Press the enter button to stop.]', end='')
    
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    input('')
    STOP_FLAG.set()
    recording_thread.join()
    STOP_FLAG.clear()

    wf = wave.open(WAVE_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(SHARED_QUEUE.get()))
    wf.close()

    with open(WAVE_FILE, 'rb') as fi:
        response = openai.Audio.transcribe("whisper-1", fi)

    os.remove(WAVE_FILE)

    return response['text']

### Chatbot conversation helper variables and functions ###

SYSTEM_INITIALIZATIONS = {
    'Default': re.sub('[\t\r\n]+', '', f'''
        You are ChatGPT, a large language model trained by OpenAI.
        Answer as concisely as possible.
        Knowledge cutoff: {KNOWLEDGE_CUTOFF}
        Current date: {CURRENT_DATE}
        '''),
    'Helpful': re.sub('[\t\r\n]+', '', '''
        You are a helpful assistant.
        '''),
    'Great Depth': re.sub('[\t\r\n]+', '', '''
        You are a friendly and helpful teaching assistant. 
        You explain concepts in great depth using simple terms, and you give examples to help people learn.
        At the end of each explanation, you ask a question to check for understanding
        '''),
    'Laconic': re.sub('[\t\r\n]+', '', '''
        You are a laconic assistant. You reply with brief, to-the-point answers with no elaboration.
        '''),
}

def setup_gpt():
    global CONTEXT
    global HISTORY
    
    CONTEXT = [{'role': 'system', 'content': SYSTEM_INITIALIZATIONS['Default']}]
    HISTORY = []
    
    print('')
    print('Starting a new conversation.')
    print('Input "new" to begin a new conversation.')
    print('Input "exit" to terminate the conversation.')
    print('')
    
def ask_gpt(query):
    global CONTEXT
    global HISTORY
    
    CONTEXT.append({'role': 'user', 'content': query})
    
    new_input = {'model': MODEL, 'messages': CONTEXT}
    new_output = openai.ChatCompletion.create(**new_input)
    new_output_text = new_output['choices'][0]['message']['content']
    
    HISTORY.append({'input': new_input, 'output': new_output})
    CONTEXT.append({'role': 'assistant', 'content': new_output_text})
    
    return new_output_text

### Start conversation loop ###

setup_gpt()
while True:
    if AUDIO_QUERIES:
        print('User: ')
        query = get_audio_query()
        print(query.strip())
    else:
        query = input('User: ')
    
    if query.strip().lower() in ['new', 'new.', 'new!']:
        setup_gpt()
        continue
    
    if query.strip().lower() in ['exit', 'exit.', 'exit!']:
        break
    
    response = ask_gpt(query)
    print('\nAssistant:', response, '\n')


