from dataPreprocess import apply_specaug, get_melspec, spec_to_image
import model
import torch
import pl_model
import os
import torchinfo

model = pl_model.MosquitoClassifier.load_from_checkpoint(r"D:\ICS Project\Project\checkpoints\resnet-attention-epoch=16-val_acc=0.86.ckpt")

def predict(path):
        
        audio = get_melspec(path)

        audio = spec_to_image(audio)

        audio = apply_specaug(audio)

        audio = torch.from_numpy(audio)
        audio = audio.unsqueeze(0)
        torchinfo.summary(model, input_data=audio, depth=7)
        # print(audio.shape)
        result = model(audio)
        # print(result)

        preds = torch.argmax(result, dim=1).item()

        label_list = {
                0: "Ae. aegypti", 
                1: "Ae. albopictus", 
                2: "An. arabiensis", 
                3: "An. gambiae", 
                4: "C. quinquefasciatus", 
                5: "C. pipiens"
            }

        return label_list[preds]

