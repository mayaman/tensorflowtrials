sound_file_path = "/audio/fold1/"

sound_file_paths = ["7061-6-0-0.wav", "7383-3-0-0.wav", "17592-5-1-1.wav", "24074-1-0-9.wav", "21684-9-0-5.wav", "146186-5-0-11.wav"]

sound_names = ["gun shot","dog bark", "jackhammer", "siren", "street music", "engine idling"]

for sf in sound_file_paths:
    sf = sound_file_path + sf

raw_sounds = load_sound_files(sound_file_paths)

plot_waves(sound_names,raw_sounds)
plot_specgram(sound_names,raw_sounds)
plot_log_power_specgram(sound_names,raw_sounds)
