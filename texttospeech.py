# from transformers import pipeline
# from datasets import load_dataset
# import soundfile as sf
# import torch


# synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# # You can replace this embedding with your own as well.

# speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

# sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])


from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="speaker.wav",
                language="en")
