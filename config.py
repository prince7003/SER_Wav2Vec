class Config:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-large-960h-lv60-self"
        self.num_labels = 8
        self.batch_size = 16
        self.learning_rate = 3e-5
        self.num_epochs = 30
        self.max_duration = 5
        self.sampling_rate = 16000
        self.output_dir = "./results"
        self.seed = 42
        self.ravdess_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        self.ravdess_dir = "./ravdess_data"

config = Config()