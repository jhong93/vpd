from contextlib import nullcontext
from collections import Counter
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

from .util import step, batch_mulitplexer, batch_zipper


TRIPLET_LOSS = False


class _BaseModel:

    def __init__(self, encoder, decoders, device):
        self.encoder = encoder
        self.decoders = decoders
        self.device = device

        # Move to device
        self.encoder.to(device)
        for decoder in self.decoders.values():
            decoder.to(device)

    def _train(self):
        self.encoder.train()
        for decoder in self.decoders.values():
            decoder.train()

    def _eval(self):
        self.encoder.eval()
        for decoder in self.decoders.values():
            decoder.eval()


class Keypoint_EmbeddingModel(_BaseModel):

    def epoch(self, data_loaders, optimizer=None, scaler=None, progress_cb=None,
              weight_3d=1):
        self._train() if optimizer is not None else self._eval()

        dataset_losses = Counter()
        dataset_contra_losses = Counter()
        dataset_counts = Counter()
        with torch.no_grad() if optimizer is None else nullcontext():
            for zipped_batch in (
                batch_zipper(data_loaders) if optimizer is not None
                else ((x,) for x in batch_mulitplexer(data_loaders))
            ):
                batch_loss = 0.
                batch_n = 0

                for dataset_name, batch in zipped_batch:
                    contra_loss = 0.

                    with nullcontext() if scaler is None else amp.autocast():
                        pose1 = batch['pose1'].to(self.device)
                        n = pose1.shape[0]
                        emb1 = self.encoder(pose1.view(n, -1))
                        emb2 = None
                        if 'pose2' in batch:
                            pose2 = batch['pose2'].to(self.device)
                            emb2 = self.encoder(pose2.view(n, -1))
                            contra_loss += F.hinge_embedding_loss(
                                torch.norm(emb1 - emb2, dim=1),
                                torch.ones(n, dtype=torch.int32, device=self.device),
                                reduction='sum')

                        if 'pose_neg' in batch:
                            pose_neg = batch['pose_neg'].to(self.device)
                            emb_neg = self.encoder(pose_neg.view(n, -1))

                            if TRIPLET_LOSS:
                                contra_loss += torch.sum(F.triplet_margin_with_distance_loss(
                                    emb1, emb2, emb_neg, reduction='none'
                                ) * batch['pose_neg_is_valid'].to(self.device))
                            else:
                                contra_loss += torch.sum(F.hinge_embedding_loss(
                                torch.norm(emb1 - emb_neg, dim=1),
                                    -torch.ones(n, dtype=torch.int32, device=self.device),
                                    reduction='none'
                                ) * batch['pose_neg_is_valid'].to(self.device))

                        loss = 0.
                        loss += contra_loss

                        if 'kp_features' in batch:
                            pose_decoder = self.decoders['3d']
                            true3d = batch['kp_features'].float().to(self.device)

                            pred3d1 = pose_decoder(
                                emb1, dataset_name).reshape(true3d.shape)
                            loss += weight_3d * F.mse_loss(
                                pred3d1, true3d, reduction='sum')

                            if emb2 is not None:
                                pred3d2 = pose_decoder(
                                    emb2, dataset_name).reshape(true3d.shape)
                                loss += weight_3d * F.mse_loss(
                                    pred3d2, true3d, reduction='sum')

                    if contra_loss > 0:
                        dataset_contra_losses[dataset_name] += contra_loss.item()
                    dataset_losses[dataset_name] += loss.item()
                    dataset_counts[dataset_name] += n

                    # Sum the losses for the dataset
                    batch_loss += loss
                    batch_n += n

                # Take mean of losses before backprop
                batch_loss /= batch_n

                if optimizer is not None:
                    step(optimizer, scaler, batch_loss)

                if progress_cb is not None:
                    progress_cb(batch_n)

        # print({k: v / dataset_counts[k]
        #        for k, v in dataset_contra_losses.items()})

        epoch_n = sum(dataset_counts.values())
        return (sum(dataset_contra_losses.values()) / epoch_n,
                sum(dataset_losses.values()) / epoch_n,
                {k: v / dataset_counts[k] for k, v in dataset_losses.items()})

    def _predict(self, pose, get_emb, decoder_target=None):
        assert get_emb or decoder_target is not None, 'Nothing to predict'
        if not isinstance(pose, torch.Tensor):
            pose = torch.FloatTensor(pose)

        pose = pose.to(self.device)
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(0)

        self.encoder.eval()
        if decoder_target is not None:
            decoder = self.decoders['3d']
            decoder.eval()
        else:
            decoder = None

        with torch.no_grad():
            n = pose.shape[0]
            emb = self.encoder(pose.view(n, -1))
            if decoder is None:
                return emb.cpu().numpy(), None

            pred3d = decoder(emb, decoder_target)
            if get_emb:
                return emb.cpu().numpy(), pred3d.cpu().numpy()
            else:
                return None, pred3d.cpu().numpy()

    def embed(self, pose):
        return self._predict(pose, get_emb=True)[0]

    def predict3d(self, pose, decoder_target):
        return self._predict(
            pose, get_emb=False, decoder_target=decoder_target)[1]

    def embed_and_predict3d(self, pose, decoder_target):
        return self._predict(
            pose, get_emb=True, decoder_target=decoder_target)