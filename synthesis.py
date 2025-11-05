from melo.api import TTS
import torch
import nltk
def get_device() -> str:
    """Auto-detect torch device: MPS (Apple M1), CUDA, or CPU."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda:0'

    return 'cpu'

device = get_device()
print(device)
text = "This is proof that Melotts works within a Google Collab environment."

model = TTS(language="EN", device=device) # create the TTS model

# dialect 
speaker_key = 'EN-BR'
speaker_id = model.hps.data.spk2id[speaker_key]

# file key is somewhat similar to the speaker key, 
# it identifies the location of the weight (for a particular
# dialect. The below is simple string manipulation.
file_key = speaker_key.lower().replace('_', '-')

# loading the base speaker weight
source_se = torch.load(f"/content/OpenVoice/checkpoints_v2/base_speakers/ses/{file_key}.pth", map_location=device) 

# save the synthesised text into a temp file. 
model.tts_to_file(text, speaker_id, "/content/tmp.wav", speed=1)
