from openvoice.api import ToneColorConverter

converter = ToneColorConverter(
        config_path="./checkpoints_v2/converter/config.json",
        device=device,
    )
converter.load_ckpt("./checkpoints_v2/converter/checkpoint.pth")

from openvoice import se_extractor
# C:\Users\mp\Documents\athens\OpenVoice\myvoice\anne_of_avonlea_01_montgomery.mp3
target_se, _ = se_extractor.get_se(str("./myvoice/anne_of_avonlea_01_montgomery.mp3"), converter, vad=True)

with torch.no_grad():
    converter.convert(
        audio_src_path="./tmp.wav",
        src_se=source_se,
        tgt_se=target_se,
        output_path="./new.wav",
        message="@MyShell", #a voice identier
    )