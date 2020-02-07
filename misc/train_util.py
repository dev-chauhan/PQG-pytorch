import torch
import misc.utils as utils
import misc.net_utils as net_utils
import torch.nn as nn
from pycocoevalcap.eval import COCOEvalCap
import os


def getObjsForScores(real_sents, pred_sents):
    class coco:
        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption': sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]

    return coco(real_sents), coco(pred_sents)


def evaluate_scores(s1, s2):

    '''
    calculates scores and return the dict with score_name and value
    '''
    coco, cocoRes = getObjsForScores(s1, s2)

    evalObj = COCOEvalCap(coco, cocoRes)

    evalObj.evaluate()

    return evalObj.eval


def dump_samples(ph, pph, gpph, file_name):

    file = open(file_name, "w")

    for r, s, t in zip(ph, pph, gpph):
        file.write("ph : " + r + "\npph : " + s + "\ngpph : " + t + '\n\n')
    file.close()


# def save_model(encoder, generator, model_optim, epoch, it, local_loss, global_loss, save_folder, folder, discriminator=None, discriminatorg=None):

#     PATH = os.path.join(save_folder, folder, str(epoch) + '_' + str(it) + '.tar')

#     checkpoint = {
#         'epoch': epoch,
#         'iter': it,
#         'encoder_state_dict': encoder.state_dict(),
#         'generator_state_dict': generator.state_dict(),
#         'optimizer_state_dict': model_optim.state_dict(),
#         'local_loss': local_loss,
#         'global_loss': global_loss
#     }
#     if discriminator is not None:
#         checkpoint['discriminator_state_dict'] = discriminator.state_dict()
#     if discriminatorg is not None:
#         checkpoint['discriminatorg_state_dict'] = discriminatorg.state_dict()

#     torch.save(checkpoint, PATH)

def save_model(model, model_opt, epoch, save_file):

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'model_opt': model_opt.state_dict()
    }

    torch.save(checkpoint, save_file)
