from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

model_output, midi_data, note_events = predict('/home/martins_o/HackNote/17573__danglada__g-major.wav')
print(model_output)