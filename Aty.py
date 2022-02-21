#Chatbot
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#Speech To text
import speech_recognition as sr 

#Text to speech
from gtts import gTTS
from playsound import playsound

from curtsies.fmtfuncs import red, bold, green, on_blue, yellow, blue, cyan


TOKENIZERS_PARALLELISM= True


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to("cuda")

r = sr.Recognizer()


def speak_text(text,lang='en'):
    tts = gTTS(text,lang=lang)
    tts.save('out.mp3')
    playsound('out.mp3')


with sr.Microphone() as source:
        step = 0
        # read the audio data from the default microphone
        print("Calibrating audio")
        r.adjust_for_ambient_noise(source, duration=3)
        print("Calibrated, you can talk now")
        

for step in range(100):
    with sr.Microphone() as source:
      
        
        audio_data = r.listen(source)
       
        text = r.recognize_google(audio_data)
        mic_in = str(text)
        print(green(mic_in))

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(mic_in + tokenizer.eos_token, return_tensors='pt').to("cuda")

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, temperature=0.9).to("cuda")

    # pretty print last ouput tokens from bot
    output_text = "{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    output = "Aty: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    print(red(output)) 
    speak_text(output_text)





