from pyctcdecode import build_ctcdecoder

class NgramDecoder:
    def __init__(self,vocab_dict, lm_model_path, alpha=0.5, beta=1.0):
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        self.labels = list(sorted_vocab_dict.keys())
        self.decoder = build_ctcdecoder(labels=self.labels,
                                         kenlm_model_path=lm_model_path,  # either .arpa or .bin file
                                         alpha=alpha,  
                                         beta=beta,  
                                       )
    def decode(self, logits, hotwords = None, hotword_weight=10.0):
        x = logits.squeeze()
        x = x.cpu().detach().numpy()
        text = self.decoder.decode(x,
                     hotwords=hotwords,
                     hotword_weight=hotword_weight)

        return text

