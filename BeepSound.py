from playsound import playsound

times = 2
frequency = 1000  # Frequency is not used with playsound
duration = 1000  # Duration is not used with playsound

def beepsound(sound_file_path):
    for _ in range(times):
        playsound(sound_file_path)
