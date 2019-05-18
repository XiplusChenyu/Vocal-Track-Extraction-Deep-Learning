import torch
import librosa
import numpy as np

from mel_dealer import mel_converter
from utils import embedding_to_mask
from config import PARAS
from utils import mask_scale_loss_unet


class VocalExtractor:
    """
    This create a vocal extractor class
    """
    def __init__(self, model, weight_name, model_type):
        """
        model type should either be mask, or cluster
        :param model: use certain model here
        :param weight_name: get model path here
        :param model_type:
        """
        self.model = model
        self.model.load_state_dict(torch.load(PARAS.MODEL_SAVE_PATH + weight_name, map_location='cpu'))
        self.model.eval()
        self.type = model_type

    def get_cluster_out(self, inp):
        inp = torch.Tensor(inp)
        shape = inp.shape
        with torch.no_grad():
            inp = inp.view((1, shape[0], -1))
            predict = self.model(inp)
        embedding, mask = predict
        mask, _ = torch.unbind(mask.squeeze(0), dim=2)
        emb1, emb2 = embedding_to_mask(embedding)
        out_mask = emb2 if mask_scale_loss_unet(torch.Tensor(emb1), mask) > \
                           mask_scale_loss_unet(torch.Tensor(emb2), mask) else emb1
        return out_mask

    def get_simple_cluster_out(self, inp):
        inp = torch.Tensor(inp)
        shape = inp.shape
        with torch.no_grad():
            inp = inp.view((1, shape[0], -1))
            predict = self.model(inp)
        emb1, emb2 = embedding_to_mask(predict)
        return emb1, emb2

    def get_unet_out(self, inp):
        inp = torch.Tensor(inp)
        with torch.no_grad():
            inp = inp.unsqueeze(0).unsqueeze(1)
            predicted = self.model(inp)
        return predicted.squeeze(0).numpy()

    def apply_mask(self, chunk, mask):
        chunk_amplitude = librosa.db_to_power(chunk)
        masked = chunk_amplitude * mask
        return masked

    def gain_chunk_result(self, chunk):
        """
        Takes in mel spectrogram, output mask (scale mask for unet, binary mask for dchd)
        :param chunk:
        :return:
        """
        if self.type == 'cluster':
            mask = self.get_cluster_out(chunk)
        elif self.type == 'mask':
            mask = self.get_unet_out(chunk)
        else:
            raise Exception('type should be cluster or mask')
        masked = self.apply_mask(chunk, mask)
        return mel_converter.melspec_to_audio(masked, log=False, audio_out=False)

    @staticmethod
    def convert_music(signal):
        S = mel_converter.signal_to_melspec(signal)
        if not S.shape[0] % PARAS.N_MEL == 0:
            S = S[:-1 * (S.shape[0] % PARAS.N_MEL)]  # divide the mel spectrogram
        chunk_num = int(S.shape[0] / PARAS.N_MEL)
        mel_chunks = np.split(S, chunk_num)  # create data frames
        return mel_chunks

    def vocal_extractor(self, signal):
        """
        takes in a music file and generate the vocal track
        :param signal: signal should be chunked
        :return:
        """
        mel_chunks = self.convert_music(signal)
        vocal_sequence = list()
        for chunk in mel_chunks:
            sub_sequence = list(self.gain_chunk_result(chunk))
            vocal_sequence += sub_sequence
        return np.array(vocal_sequence)
